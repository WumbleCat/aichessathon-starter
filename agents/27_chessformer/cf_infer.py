"""Runtime inference for the Chessformer: a pure-numpy forward pass (no torch on the clock).

The shipped file models/chessformer.pt holds {"config": {...}, "state_dict": {name: tensor}} as
written by training/train.py. torch is only used to read the file at import time; every move-time
call runs in numpy, which at batch 1 beats torch's per-op overhead on one core.

Numerically identical to cf_model.Chessformer (see tests/test_model.py); the differences are
purely algebraic re-arrangements that cut the number of numpy calls:
  * each LayerNorm's affine (w, b) is folded into the Linear layers that consume it;
  * the attention scale is folded into the query weights;
  * the constant parts of the attention bias (geometric bias + shared projection bias) are
    pre-summed into one [H, 64, 64] tensor per layer.
"""

import math

import chess
import numpy as np

from cf_encode import NUM_MOVES, encode, geometry_tables, move_index, promo_tables

_LN_EPS = 1e-5
_SQRT_2_OVER_PI = 0.7978845608028654


def _gelu(x: np.ndarray) -> np.ndarray:
    # tanh approximation, matching nn.GELU(approximate="tanh")
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    np.tanh(inner, out=inner)
    inner += 1.0
    inner *= x
    inner *= 0.5
    return inner


def _normalize(x: np.ndarray) -> np.ndarray:
    """LayerNorm without the affine part (the affine is folded into the next linear)."""
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    var = np.mean(centred * centred, axis=-1, keepdims=True)
    var += _LN_EPS
    np.sqrt(var, out=var)
    centred /= var
    return centred


def _fold_ln(w_ln: np.ndarray, b_ln: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LN(x) @ w + b  ==  norm(x) @ (diag(w_ln) w) + (b_ln @ w + b)."""
    return (w_ln[:, None] * w).astype(np.float32), (b_ln @ w + b).astype(np.float32)


class NumpyChessformer:
    def __init__(self, config: dict[str, int], state: dict[str, np.ndarray]) -> None:
        self.cfg = config
        d = config["dim"]
        self.heads = config["heads"]
        self.head_dim = d // self.heads
        scale = 1.0 / math.sqrt(self.head_dim)
        f32 = lambda k: np.ascontiguousarray(state[k].astype(np.float32))  # noqa: E731
        lin = lambda k: (f32(k + ".weight").T.copy(), f32(k + ".bias"))  # noqa: E731
        self.embed_w = f32("embed.weight").T.copy()
        self.embed_b = (f32("embed.bias")[None, :] + f32("square_embed")).astype(np.float32)
        smol_out_w, smol_out_b = lin("smol_out")
        tables = geometry_tables()
        self.layers = []
        for i in range(config["layers"]):
            p = f"blocks.{i}."
            geo = (
                state[p + "geo.dx"][:, tables["dx"]]
                + state[p + "geo.dy"][:, tables["dy"]]
                + state[p + "geo.rel"][:, tables["rel"]]
                + state[p + "geo.dist"][:, tables["dist"]]
            ).astype(np.float32)  # [H, 64, 64]
            ln1_w, ln1_b = f32(p + "ln1.weight"), f32(p + "ln1.bias")
            qkv_w, qkv_b = _fold_ln(ln1_w, ln1_b, *lin(p + "qkv"))
            # fold the attention scale into the query third
            qkv_w[:, :d] *= scale
            qkv_b[:d] *= scale
            cmp_w = (ln1_w[:, None] * f32(p + "dyn.compress.weight").T).astype(np.float32)
            cmp_b = (ln1_b @ f32(p + "dyn.compress.weight").T).astype(np.float32)  # [smol]
            hid_w, hid_b = lin(p + "dyn.hidden")
            # compress bias is the same for every token: fold it into the hidden bias
            hid_b = (np.tile(cmp_b, 64) @ hid_w + hid_b).astype(np.float32)
            dn_w, dn_b = f32(p + "dyn.norm.weight"), f32(p + "dyn.norm.bias")
            out_w, out_b = _fold_ln(dn_w, dn_b, smol_out_w, smol_out_b)
            ln2_w, ln2_b = f32(p + "ln2.weight"), f32(p + "ln2.bias")
            mlp1_w, mlp1_b = _fold_ln(ln2_w, ln2_b, *lin(p + "mlp.0"))
            mlp2_w, mlp2_b = lin(p + "mlp.2")
            proj_w, proj_b = lin(p + "proj")
            self.layers.append(
                {
                    "qkv_w": qkv_w,
                    "qkv_b": qkv_b,
                    "proj_w": proj_w,
                    "proj_b": proj_b,
                    "bias": np.ascontiguousarray(geo + out_b.reshape(self.heads, 64, 64)),
                    "cmp_w": np.ascontiguousarray(cmp_w),
                    "hid_w": hid_w,
                    "hid_b": hid_b,
                    "out_w": np.ascontiguousarray(out_w),
                    "mlp1_w": mlp1_w,
                    "mlp1_b": mlp1_b,
                    "mlp2_w": mlp2_w,
                    "mlp2_b": mlp2_b,
                }
            )
        lnf_w, lnf_b = f32("ln_f.weight"), f32("ln_f.bias")
        pol_scale = 1.0 / math.sqrt(config["policy_dim"])
        self.pq_w, self.pq_b = _fold_ln(lnf_w, lnf_b, *lin("pol_q"))
        self.pq_w *= pol_scale
        self.pq_b *= pol_scale
        self.pk_w, self.pk_b = _fold_ln(lnf_w, lnf_b, *lin("pol_k"))
        self.pp_w, self.pp_b = _fold_ln(lnf_w, lnf_b, *lin("pol_promo"))
        self.v1_w, self.v1_b = _fold_ln(lnf_w, lnf_b, *lin("val.0"))
        self.v2_w, self.v2_b = lin("val.2")
        base, dst, piece = promo_tables()
        self.promo_base = base
        self.promo_flat = dst * 4 + piece  # index into promo.reshape(-1)

    def forward(self, feats: np.ndarray) -> tuple[np.ndarray, float]:
        """feats [64, 19] -> (policy logits [NUM_MOVES], value in (-1, 1))."""
        h = self.heads
        hd = self.head_dim
        x = feats @ self.embed_w
        x += self.embed_b
        for L in self.layers:  # noqa: N806
            n = _normalize(x)
            qkv = n @ L["qkv_w"]
            qkv += L["qkv_b"]
            qkv = qkv.reshape(64, 3, h, hd).transpose(1, 2, 0, 3)  # [3, H, 64, hd]
            logits = np.matmul(qkv[0], qkv[1].transpose(0, 2, 1))  # [H, 64, 64]
            logits += L["bias"]
            c = (n @ L["cmp_w"]).reshape(-1)
            g = c @ L["hid_w"]
            g += L["hid_b"]
            g = _normalize(_gelu(g))
            logits += (g @ L["out_w"]).reshape(h, 64, 64)
            logits -= logits.max(axis=-1, keepdims=True)
            np.exp(logits, out=logits)
            logits /= logits.sum(axis=-1, keepdims=True)
            out = np.matmul(logits, qkv[2]).transpose(1, 0, 2).reshape(64, h * hd)
            x += out @ L["proj_w"]
            x += L["proj_b"]
            n = _normalize(x)
            m = n @ L["mlp1_w"]
            m += L["mlp1_b"]
            x += _gelu(m) @ L["mlp2_w"]
            x += L["mlp2_b"]
        n = _normalize(x)
        q = n @ self.pq_w
        q += self.pq_b
        k = n @ self.pk_w
        k += self.pk_b
        base = (q @ k.T).reshape(4096)
        promo = n @ self.pp_w
        promo += self.pp_b
        policy = np.empty(NUM_MOVES, dtype=np.float32)
        policy[:4096] = base
        policy[4096:] = base[self.promo_base] + promo.reshape(-1)[self.promo_flat]
        pooled = n.mean(axis=0)
        v = pooled @ self.v1_w
        v += self.v1_b
        value = float(np.tanh(_gelu(v) @ self.v2_w + self.v2_b)[0])
        return policy, value


def load_state(path: str) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    if path.endswith(".npz"):
        import json

        z = np.load(path)
        config = json.loads(str(z["__config__"]))
        state = {k: z[k] for k in z.files if k != "__config__"}
        return config, state
    import torch

    torch.set_num_threads(1)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    config = {k: int(v) for k, v in blob["config"].items()}
    state = {k: v.detach().cpu().numpy() for k, v in blob["state_dict"].items()}
    return config, state


class PolicyModel:
    """Move priors and a value for a python-chess board, from the numpy Chessformer."""

    def __init__(self, path: str) -> None:
        config, state = load_state(path)
        self.net = NumpyChessformer(config, state)
        self.config = config
        self._feats = np.zeros((64, 19), dtype=np.float32)
        self.calls = 0

    def raw(self, board: chess.Board) -> tuple[np.ndarray, float]:
        encode(board, self._feats)
        return self.net.forward(self._feats)

    def priors(self, board: chess.Board, temperature: float = 1.0) -> dict[chess.Move, float]:
        self.calls += 1
        logits, _ = self.raw(board)
        mirror = not board.turn
        moves = list(board.legal_moves)
        if not moves:
            return {}
        idx = [move_index(m, mirror) for m in moves]
        sel = logits[idx] / temperature
        sel = np.exp(sel - sel.max())
        sel /= sel.sum()
        return dict(zip(moves, sel.tolist(), strict=True))

    def value(self, board: chess.Board) -> float:
        return self.raw(board)[1]

    def warm_up(self) -> None:
        self.priors(chess.Board())


assert NUM_MOVES == 4192
