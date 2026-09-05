"""Chessformer-specific tests: tokens, move index uniqueness, flips, numpy/torch parity."""

import os
import sys

import chess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cf_encode import NUM_FEATURES, NUM_MOVES, encode, index_move, move_index  # noqa: E402
from cf_infer import NumpyChessformer  # noqa: E402
from cf_model import Chessformer, Config  # noqa: E402

FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R b KQ - 0 8",
    "8/P7/8/8/8/8/8/k6K w - - 0 1",
    "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "r3k2r/pppqppbp/2np1np1/8/2BPP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppqppbp/2np1np1/8/2BPP3/2N1BN2/PPPQ1PPP/R3K2R b KQkq - 0 1",
]


def test_exactly_64_tokens_and_one_piece_class_per_square():
    for fen in FENS:
        f = encode(chess.Board(fen))
        assert f.shape == (64, NUM_FEATURES)
        assert np.all(f[:, :13].sum(axis=1) == 1.0)


def test_move_index_is_unique_and_invertible_for_all_legal_moves():
    seen_total = 0
    for fen in FENS:
        board = chess.Board(fen)
        mirror = not board.turn
        idx = {}
        for m in board.legal_moves:
            i = move_index(m, mirror)
            assert 0 <= i < NUM_MOVES
            assert i not in idx, f"collision {m} / {idx[i]} in {fen}"
            idx[i] = m
            assert index_move(i, mirror) == m
            seen_total += 1
    assert seen_total > 100


def test_all_four_promotions_map_uniquely():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    ids = {move_index(m, False) for m in board.legal_moves if m.promotion}
    assert len(ids) == 4
    assert all(i >= 4096 for i in ids)
    # capture-promotions to a neighbouring file get their own slots too
    board = chess.Board("1n6/P7/8/8/8/8/8/k6K w - - 0 1")
    ids = {move_index(m, False) for m in board.legal_moves if m.promotion}
    assert len(ids) == 8


def test_colour_normalisation_makes_mirrored_positions_identical():
    """A position and its colour-mirror must produce identical tokens and identical outputs."""
    torch.manual_seed(0)
    model = Chessformer(Config(dim=32, layers=1, heads=2, smol_hidden=16)).eval()
    for fen in FENS:
        board = chess.Board(fen)
        mirrored = board.mirror()
        a = encode(board)
        b = encode(mirrored)
        assert np.array_equal(a, b), fen
        with torch.no_grad():
            pa, va = model(torch.from_numpy(a).unsqueeze(0))
            pb, vb = model(torch.from_numpy(b).unsqueeze(0))
        assert torch.allclose(pa, pb) and torch.allclose(va, vb)
        # and the legal move sets map onto the same index set
        ia = sorted(move_index(m, not board.turn) for m in board.legal_moves)
        ib = sorted(move_index(m, not mirrored.turn) for m in mirrored.legal_moves)
        assert ia == ib


def test_geometric_bias_indices_are_symmetric_under_flip():
    from cf_encode import geometry_tables

    t = geometry_tables()
    for s in range(64):
        for d in range(64):
            fs, fd = s ^ 56, d ^ 56
            swap = {4: 5, 5: 4}
            assert t["rel"][s, d] == swap.get(t["rel"][fs, fd], t["rel"][fs, fd])
            assert t["dist"][s, d] == t["dist"][fs, fd]
            assert t["dx"][s, d] == t["dx"][fs, fd]
            assert t["dy"][s, d] + t["dy"][fs, fd] == 14


def test_numpy_forward_matches_torch():
    torch.manual_seed(1)
    cfg = Config(dim=64, layers=2, heads=4, smol_hidden=32)
    model = Chessformer(cfg).eval()
    # perturb parameters so biases are not near zero
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    net = NumpyChessformer(cfg.as_dict(), state)
    for fen in FENS:
        f = encode(chess.Board(fen))
        with torch.no_grad():
            pt, vt = model(torch.from_numpy(f).unsqueeze(0))
        pn, vn = net.forward(f)
        assert np.allclose(pt[0].numpy(), pn, atol=1e-4), fen
        assert abs(float(vt[0]) - vn) < 1e-4


def test_legal_mask_priors_sum_to_one():
    from cf_infer import PolicyModel

    torch.manual_seed(2)
    cfg = Config(dim=32, layers=1, heads=2, smol_hidden=16)
    model = Chessformer(cfg)
    path = os.path.join(HERE, "_tmp_model.pt")
    torch.save({"config": cfg.as_dict(), "state_dict": model.state_dict()}, path)
    try:
        pm = PolicyModel(path)
        for fen in FENS:
            board = chess.Board(fen)
            pri = pm.priors(board)
            assert set(pri) == set(board.legal_moves)
            assert abs(sum(pri.values()) - 1.0) < 1e-5
    finally:
        os.remove(path)


def test_compact_expand_matches_encode():
    from cf_encode import compact, expand

    rows = np.stack([compact(chess.Board(f)) for f in FENS])
    full = expand(rows)
    for i, fen in enumerate(FENS):
        assert np.array_equal(full[i], encode(chess.Board(fen))), fen
