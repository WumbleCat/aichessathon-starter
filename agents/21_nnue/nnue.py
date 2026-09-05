"""NNUE evaluator: (768 -> H) x 2 perspectives -> 1, int16 accumulators, incremental update.

Feature index for perspective ``persp`` (0 = white, 1 = black) of a piece with colour ``c``
and type ``t`` (1..6) on square ``sq``::

    rel_colour = c ^ persp                     (0 = own piece)
    rel_sq     = sq ^ (56 * persp)             (vertical flip for the black perspective)
    index      = (rel_colour * 6 + t - 1) * 64 + rel_sq

Quantisation (fixed at training/export time, see training/export.py):
    W1, B1 scaled by QA (accumulator units), W2 scaled by QB, B2 by QA * QB.
    eval_cp = (sum(W2 * crelu(acc)) + B2) * SCALE / (QA * QB)

The accumulator for ply ``k`` lives in ``acc[k, persp, :]``; make_move copies and patches it
into ``acc[k + 1]`` using the P[LAST_*] slots written by cboard.make_move.

Weights ship as a ``.safetensors`` file read by a tiny local parser (no torch at runtime).
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

import cboard as cb
from jitconf import jit

NUM_FEATURES = 768
QA = 255
QB = 64
SCALE = 400

_DTYPES = {"I16": np.int16, "I32": np.int32, "F32": np.float32, "I8": np.int8, "I64": np.int64}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
        data = f.read()
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        arr = np.frombuffer(data[start:end], dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])
        out[name] = np.ascontiguousarray(arr)
    return out


def write_safetensors(
    path: str, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None
) -> None:
    inv = {v: k for k, v in _DTYPES.items()}
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        header[name] = {
            "dtype": inv[arr.dtype.type],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def load_net(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (W1 int16[768,H], B1 int16[H], W2 int16[2H], B2 int32[1])."""
    t = read_safetensors(path)
    W1 = t["W1"].astype(np.int16)
    B1 = t["B1"].astype(np.int16)
    W2 = t["W2"].astype(np.int16)
    B2 = t["B2"].astype(np.int32).reshape(1)
    assert W1.shape[0] == NUM_FEATURES and W2.shape[0] == 2 * W1.shape[1]
    return W1, B1, W2, B2


def random_net(
    hidden: int = 256, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A random network of the right shape, for tests that only need the plumbing."""
    rng = np.random.default_rng(seed)
    W1 = rng.integers(-40, 40, size=(NUM_FEATURES, hidden), dtype=np.int16)
    B1 = rng.integers(-40, 40, size=hidden, dtype=np.int16)
    W2 = rng.integers(-40, 40, size=2 * hidden, dtype=np.int16)
    B2 = np.array([0], dtype=np.int32)
    return W1, B1, W2, B2


def new_acc(hidden: int) -> np.ndarray:
    return np.zeros((cb.MAX_PLY + 2, 2, hidden), dtype=np.int16)


@jit
def feature_index(persp, piece, sq):  # type: ignore[no-untyped-def]
    c = 1 if piece >= 7 else 0
    t = piece - 6 if piece >= 7 else piece
    return ((c ^ persp) * 6 + t - 1) * 64 + (sq ^ (56 * persp))


@jit
def refresh(acc, ply, P, W1, B1):  # type: ignore[no-untyped-def]
    """Full recompute of both perspectives at ``ply`` from the board in ``P``."""
    H = W1.shape[1]
    for persp in range(2):
        for i in range(H):
            acc[ply, persp, i] = B1[i]
        for sq in range(64):
            pc = P[sq]
            if pc != 0:
                f = feature_index(persp, pc, sq)
                for i in range(H):
                    acc[ply, persp, i] += W1[f, i]


@jit
def update(acc, ply, P, W1):  # type: ignore[no-untyped-def]
    """acc[ply+1] = acc[ply] patched with the move recorded in P[LAST_*]."""
    H = W1.shape[1]
    pc = P[cb.LAST_PIECE]
    frm = P[cb.LAST_FROM]
    to = P[cb.LAST_TO]
    cap = P[cb.LAST_CAPTURED]
    promo = P[cb.LAST_PROMO]
    rfrom = P[cb.LAST_ROOK_FROM]
    for persp in range(2):
        f_rem = feature_index(persp, pc, frm)
        f_add = feature_index(persp, promo if promo != 0 else pc, to)
        if cap != 0:
            f_cap = feature_index(persp, cap, P[cb.LAST_CAPSQ])
            for i in range(H):
                acc[ply + 1, persp, i] = (
                    acc[ply, persp, i] - W1[f_rem, i] + W1[f_add, i] - W1[f_cap, i]
                )
        elif rfrom >= 0:
            rook = cb.make_piece(cb.piece_color(pc), cb.ROOK)
            f_rr = feature_index(persp, rook, rfrom)
            f_ra = feature_index(persp, rook, P[cb.LAST_ROOK_TO])
            for i in range(H):
                acc[ply + 1, persp, i] = (
                    acc[ply, persp, i] - W1[f_rem, i] + W1[f_add, i] - W1[f_rr, i] + W1[f_ra, i]
                )
        else:
            for i in range(H):
                acc[ply + 1, persp, i] = acc[ply, persp, i] - W1[f_rem, i] + W1[f_add, i]


@jit
def copy_acc(acc, ply):  # type: ignore[no-untyped-def]
    """acc[ply+1] = acc[ply] (null move)."""
    H = acc.shape[2]
    for persp in range(2):
        for i in range(H):
            acc[ply + 1, persp, i] = acc[ply, persp, i]


@jit
def evaluate(acc, ply, side, W2, B2):  # type: ignore[no-untyped-def]
    """Centipawns from the side-to-move's point of view."""
    H = acc.shape[2]
    s = np.int64(0)
    for i in range(H):
        v = np.int64(acc[ply, side, i])
        if v < 0:
            v = np.int64(0)
        elif v > QA:
            v = np.int64(QA)
        s += v * W2[i]
    for i in range(H):
        v = np.int64(acc[ply, 1 - side, i])
        if v < 0:
            v = np.int64(0)
        elif v > QA:
            v = np.int64(QA)
        s += v * W2[H + i]
    s += B2[0]
    return (s * SCALE) // (QA * QB)


def default_weights_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "nnue.safetensors")
