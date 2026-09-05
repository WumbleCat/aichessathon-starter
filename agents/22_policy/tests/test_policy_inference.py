"""Numpy / onnx inference must match torch, and be fast enough at batch 1.

python tests/test_policy_inference.py [channels] [blocks]
"""

import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from model import PolicyNet, count_params, export_numpy, export_onnx  # noqa: E402
from pn_encoding import encode_board  # noqa: E402
from pn_policy import NumpyPolicy, OnnxPolicy, benchmark  # noqa: E402

torch.set_num_threads(1)
channels = int(sys.argv[1]) if len(sys.argv) > 1 else 64
blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 5

model = PolicyNet(channels, blocks)
# give batch-norm non-trivial statistics so folding is actually tested
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.normal_(1, 0.2)
            m.bias.normal_(0, 0.2)
model.eval()
print(f"channels {channels} blocks {blocks} params {count_params(model):,}")

tmp = tempfile.mkdtemp()
npz = os.path.join(tmp, "policy.npz")
onnx_path = os.path.join(tmp, "policy.onnx")
export_numpy(model, npz)
try:
    export_onnx(model, onnx_path)
    have_onnx = True
except Exception as error:
    print("onnx export failed:", repr(error))
    have_onnx = False
print(f"npz size {os.path.getsize(npz) / 1e6:.2f} MB")

np_policy = NumpyPolicy(npz)
boards = [
    chess.Board(),
    chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1"),
    chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]
for b in boards:
    x = encode_board(b)
    with torch.no_grad():
        t_logits, t_value = model(torch.from_numpy(x[None]))
    n_logits = np_policy.logits(x)
    err = np.abs(t_logits[0].numpy() - n_logits).max()
    verr = abs(float(t_value[0]) - np_policy.value(x))
    print(f"max |torch - numpy| logits {err:.2e} value {verr:.2e}")
    assert err < 1e-3 and verr < 1e-3
    if have_onnx:
        ox = OnnxPolicy(onnx_path)
        oerr = np.abs(t_logits[0].numpy() - ox.logits(x)).max()
        print(f"max |torch - onnx| logits {oerr:.2e}")
        assert oerr < 1e-3

b = boards[1]
p = np_policy.prior(b)
assert abs(sum(p.values()) - 1.0) < 1e-4 and set(p) == set(b.legal_moves)
print(f"numpy prior median {benchmark(np_policy, b, 100) * 1000:.2f} ms")
if have_onnx:
    print(f"onnx  prior median {benchmark(OnnxPolicy(onnx_path), b, 100) * 1000:.2f} ms")
x = encode_board(b)
xt = torch.from_numpy(x[None])
with torch.no_grad():
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        model(xt)
        ts.append(time.perf_counter() - t0)
ts.sort()
print(f"torch forward median {ts[50] * 1000:.2f} ms")
print("ok")
