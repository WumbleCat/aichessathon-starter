"""CPU-time cost of one prior() call for numpy vs torch back ends at several net sizes."""

import os
import sys
import tempfile
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import chess  # noqa: E402
import torch  # noqa: E402
from model import PolicyNet, count_params, export_numpy  # noqa: E402
from pn_encoding import encode_board  # noqa: E402
from pn_policy import NumpyPolicy  # noqa: E402

torch.set_num_threads(1)
board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1")
x = encode_board(board)
xt = torch.from_numpy(x[None])


def cpu_ms(fn, n=200):
    fn()
    c0 = time.process_time()
    w0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.process_time() - c0) * 1000 / n, (time.perf_counter() - w0) * 1000 / n


c0 = time.process_time()
enc = cpu_ms(lambda: encode_board(board))
print(f"encode_board cpu {enc[0]:.3f} ms")

for channels, blocks in [(32, 3), (48, 4), (64, 4), (64, 5), (96, 5)]:
    model = PolicyNet(channels, blocks).eval()
    tmp = os.path.join(tempfile.mkdtemp(), "p.npz")
    export_numpy(model, tmp)
    npp = NumpyPolicy(tmp)
    with torch.no_grad():
        t_cpu, t_wall = cpu_ms(lambda m=model: m(xt))
    n_cpu, n_wall = cpu_ms(lambda p=npp: p.logits(x))
    p_cpu, p_wall = cpu_ms(lambda p=npp: p.prior(board))
    print(
        f"C{channels} B{blocks} params {count_params(model):7,}  "
        f"torch cpu {t_cpu:.2f} ms (wall {t_wall:.2f})  "
        f"numpy cpu {n_cpu:.2f} ms (wall {n_wall:.2f})  numpy prior cpu {p_cpu:.2f} ms"
    )
