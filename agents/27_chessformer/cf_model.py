"""A small Chessformer-style transformer, implemented from scratch for this project.

Design (after Monroe et al., "Chessformer", ICLR 2026 and "Mastering Chess with a Transformer
Model", 2024), scaled down for one CPU core:

* the 64 squares are the tokens; each token is a linear projection of its square's features plus
  a learned square embedding;
* every attention layer adds two additive attention-logit biases per head:
  - a **geometric attention bias**: learned tables indexed by the (dx, dy) offset, the chess
    relation between source and target square (same rank / file / diagonal / anti-diagonal /
    knight jump / self) and the Chebyshev distance;
  - a **dynamic bias** generated from the whole board (the "smolgen" idea): every token is
    compressed, the compressed tokens are concatenated and fed to a small MLP whose (shared)
    final projection emits a full per-head 64x64 logit map for this position;
* an attention-style policy head: logit(from, to) = q(from) . k(to), with a per-destination
  promotion offset for N/B/R/Q so that every legal move, promotions included, has its own logit;
* a value head: mean-pooled tokens -> MLP -> scalar in (-1, 1) via tanh.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from cf_encode import (
    NUM_DELTA,
    NUM_DIST,
    NUM_FEATURES,
    NUM_MOVES,
    NUM_REL,
    geometry_tables,
    promo_tables,
)


class Config:
    def __init__(
        self,
        dim: int = 96,
        layers: int = 4,
        heads: int = 4,
        mlp_ratio: int = 2,
        smol_dim: int = 16,
        smol_hidden: int = 128,
        policy_dim: int = 64,
    ) -> None:
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.smol_dim = smol_dim
        self.smol_hidden = smol_hidden
        self.policy_dim = policy_dim

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class GeometricBias(nn.Module):
    """Learned additive attention bias from the geometric relation between two squares."""

    def __init__(self, heads: int) -> None:
        super().__init__()
        self.dx = nn.Parameter(torch.zeros(heads, NUM_DELTA))
        self.dy = nn.Parameter(torch.zeros(heads, NUM_DELTA))
        self.rel = nn.Parameter(torch.zeros(heads, NUM_REL))
        self.dist = nn.Parameter(torch.zeros(heads, NUM_DIST))
        for p in (self.dx, self.dy, self.rel, self.dist):
            nn.init.normal_(p, std=0.02)

    def forward(self, tables: dict[str, torch.Tensor]) -> torch.Tensor:
        # each gather: [H, 64, 64]
        bias = (
            self.dx[:, tables["dx"]]
            + self.dy[:, tables["dy"]]
            + self.rel[:, tables["rel"]]
            + self.dist[:, tables["dist"]]
        )
        return bias.unsqueeze(0)  # [1, H, 64, 64]


class DynamicBias(nn.Module):
    """Board-dependent attention bias ("smolgen"-style); the last projection is shared."""

    def __init__(self, cfg: Config, shared_out: nn.Linear) -> None:
        super().__init__()
        self.compress = nn.Linear(cfg.dim, cfg.smol_dim, bias=False)
        self.hidden = nn.Linear(64 * cfg.smol_dim, cfg.smol_hidden)
        self.norm = nn.LayerNorm(cfg.smol_hidden)
        self.out = shared_out  # smol_hidden -> heads * 64 * 64
        self.heads = cfg.heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.compress(x).reshape(b, -1)
        h = self.norm(F.gelu(self.hidden(h), approximate="tanh"))
        return self.out(h).reshape(b, self.heads, 64, 64)


class Block(nn.Module):
    def __init__(self, cfg: Config, shared_out: nn.Linear) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.geo = GeometricBias(cfg.heads)
        self.dyn = DynamicBias(cfg, shared_out)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * cfg.mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.dim * cfg.mlp_ratio, cfg.dim),
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, tables: dict[str, torch.Tensor]) -> torch.Tensor:
        b = x.shape[0]
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(b, 64, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, 64, hd]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        logits = logits + self.geo(tables) + self.dyn(h)
        att = torch.softmax(logits, dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(b, 64, -1)
        x = x + self.proj(out)
        x = x + self.mlp(self.ln2(x))
        return x


class Chessformer(nn.Module):
    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        cfg = cfg or Config()
        self.cfg = cfg
        self.embed = nn.Linear(NUM_FEATURES, cfg.dim)
        self.square_embed = nn.Parameter(torch.zeros(64, cfg.dim))
        nn.init.normal_(self.square_embed, std=0.02)
        self.smol_out = nn.Linear(cfg.smol_hidden, cfg.heads * 64 * 64)
        nn.init.normal_(self.smol_out.weight, std=0.01)
        nn.init.zeros_(self.smol_out.bias)
        self.blocks = nn.ModuleList(Block(cfg, self.smol_out) for _ in range(cfg.layers))
        self.ln_f = nn.LayerNorm(cfg.dim)
        # policy head
        self.pol_q = nn.Linear(cfg.dim, cfg.policy_dim)
        self.pol_k = nn.Linear(cfg.dim, cfg.policy_dim)
        self.pol_promo = nn.Linear(cfg.dim, 4)
        self.pol_scale = 1.0 / math.sqrt(cfg.policy_dim)
        # value head
        self.val = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.dim, 1),
        )
        tables = geometry_tables()
        for name, arr in tables.items():
            self.register_buffer("geo_" + name, torch.from_numpy(arr), persistent=False)
        base, dst, piece = promo_tables()
        self.register_buffer("promo_base", torch.from_numpy(base), persistent=False)
        self.register_buffer("promo_dst", torch.from_numpy(dst), persistent=False)
        self.register_buffer("promo_piece", torch.from_numpy(piece), persistent=False)

    def tables(self) -> dict[str, torch.Tensor]:
        return {
            "dx": self.geo_dx,
            "dy": self.geo_dy,
            "rel": self.geo_rel,
            "dist": self.geo_dist,
        }

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats: [B, 64, NUM_FEATURES] -> (policy logits [B, NUM_MOVES], value [B])."""
        b = feats.shape[0]
        x = self.embed(feats) + self.square_embed
        tables = self.tables()
        for block in self.blocks:
            x = block(x, tables)
        x = self.ln_f(x)
        q = self.pol_q(x)
        k = self.pol_k(x)
        base = torch.matmul(q, k.transpose(-1, -2)) * self.pol_scale  # [B, 64(from), 64(to)]
        base = base.reshape(b, 4096)
        promo = self.pol_promo(x)  # [B, 64, 4]
        promo_logits = base[:, self.promo_base] + promo[:, self.promo_dst, :].gather(
            2, self.promo_piece.view(1, -1, 1).expand(b, -1, 1)
        ).squeeze(-1)
        policy = torch.cat([base, promo_logits], dim=1)
        value = torch.tanh(self.val(x.mean(dim=1))).squeeze(-1)
        return policy, value


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(config: dict[str, int] | None = None) -> Chessformer:
    return Chessformer(Config(**config) if config else Config())


assert NUM_MOVES == 4096 + 96
