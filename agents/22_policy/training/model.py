"""Policy network definition (PyTorch, training side) and export to plain numpy weights.

Residual CNN on 18x8x8 planes -> 4672 move logits (+ an auxiliary scalar value).
The numpy inference code in ``pn_policy.py`` reads the exported dict, with batch-norm folded
into the preceding convolution.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from pn_encoding import NUM_MOVE_PLANES, NUM_PLANES  # noqa: E402


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class PolicyNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 5) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.stem = nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(blocks)])
        self.pol_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.pol_bn = nn.BatchNorm2d(channels)
        self.pol_out = nn.Conv2d(channels, NUM_MOVE_PLANES, 1, bias=True)
        self.val_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(4)
        self.val_fc1 = nn.Linear(4 * 64, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        p = torch.relu(self.pol_bn(self.pol_conv(x)))
        p = self.pol_out(p)  # (B, 73, 8, 8)
        # action index = square * 73 + plane, square = rank * 8 + file
        p = p.permute(0, 2, 3, 1).reshape(p.shape[0], 64 * NUM_MOVE_PLANES)
        v = torch.relu(self.val_bn(self.val_conv(x))).reshape(x.shape[0], -1)
        v = torch.tanh(self.val_fc2(torch.relu(self.val_fc1(v))))
        return p, v.squeeze(-1)


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d | None) -> tuple[np.ndarray, np.ndarray]:
    w = conv.weight.detach().cpu().double().numpy()
    b = conv.bias.detach().cpu().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    if bn is not None:
        gamma = bn.weight.detach().cpu().double().numpy()
        beta = bn.bias.detach().cpu().double().numpy()
        mean = bn.running_mean.detach().cpu().double().numpy()
        var = bn.running_var.detach().cpu().double().numpy()
        scale = gamma / np.sqrt(var + bn.eps)
        w = w * scale[:, None, None, None]
        b = (b - mean) * scale + beta
    return w, b


def export_numpy(model: PolicyNet, path: str) -> None:
    """Write folded weights. 3x3 convs are stored as (Cout, 9*Cin) with k = dy*3+dx major."""
    model.eval()
    arrays: dict[str, np.ndarray] = {}

    def put3(name: str, conv: nn.Conv2d, bn: nn.BatchNorm2d | None) -> None:
        w, b = _fold(conv, bn)
        cout, cin = w.shape[0], w.shape[1]
        arrays[name + "_w"] = np.ascontiguousarray(w.transpose(0, 2, 3, 1).reshape(cout, 9 * cin)).astype(np.float32)
        arrays[name + "_b"] = b.astype(np.float32)

    def put1(name: str, conv: nn.Conv2d, bn: nn.BatchNorm2d | None) -> None:
        w, b = _fold(conv, bn)
        arrays[name + "_w"] = np.ascontiguousarray(w.reshape(w.shape[0], w.shape[1])).astype(np.float32)
        arrays[name + "_b"] = b.astype(np.float32)

    put3("stem", model.stem, model.stem_bn)
    for i, block in enumerate(model.blocks):
        put3(f"block{i}_c1", block.c1, block.b1)
        put3(f"block{i}_c2", block.c2, block.b2)
    put3("pol_conv", model.pol_conv, model.pol_bn)
    put1("pol_out", model.pol_out, None)
    put1("val_conv", model.val_conv, model.val_bn)
    arrays["val_fc1_w"] = model.val_fc1.weight.detach().cpu().numpy().astype(np.float32)
    arrays["val_fc1_b"] = model.val_fc1.bias.detach().cpu().numpy().astype(np.float32)
    arrays["val_fc2_w"] = model.val_fc2.weight.detach().cpu().numpy().astype(np.float32)
    arrays["val_fc2_b"] = model.val_fc2.bias.detach().cpu().numpy().astype(np.float32)
    arrays["meta"] = np.array([model.channels, model.blocks_n], dtype=np.int32)
    np.savez(path, **arrays)


def export_onnx(model: PolicyNet, path: str) -> None:
    model.eval()
    dummy = torch.zeros(1, NUM_PLANES, 8, 8)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["planes"],
        output_names=["logits", "value"],
        dynamic_axes={"planes": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
