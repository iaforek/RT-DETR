"""Compact RT-DETR-R18 implementation for architecture study.

The structure follows the official RT-DETR PyTorch implementation:

    PResNet-18 (variant d)
        -> HybridEncoder (AIFI + CCFM/FPN-PAN)
        -> encoder top-k query selection
        -> 3-layer transformer decoder with multi-scale deformable attention
        -> 300 class/box predictions, without NMS

The code is intentionally contained in one readable file so it matches the
layout of the author's YOLOv3 and YOLOv26 learning repositories. It is trained
from random initialisation and does not load official RT-DETR weights.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Box and numerical helpers
# -----------------------------------------------------------------------------


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
        dim=-1,
    )


def bias_init_with_prob(prior_probability: float = 0.01) -> float:
    return float(-math.log((1.0 - prior_probability) / prior_probability))


def get_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(dimensions[:-1], dimensions[1:])
        )
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = self.activation(x)
        return x


# -----------------------------------------------------------------------------
# PResNet-18 backbone, matching the official RT-DETR variant-d design
# -----------------------------------------------------------------------------


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        bias: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        shortcut: bool,
        activation: str = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, ceil_mode=True)),
                            ("conv", ConvNormLayer(in_channels, out_channels, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(in_channels, out_channels, 1, stride)
        self.branch2a = ConvNormLayer(
            in_channels,
            out_channels,
            3,
            stride,
            activation=activation,
        )
        self.branch2b = ConvNormLayer(out_channels, out_channels, 3, 1)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        out = self.branch2b(self.branch2a(x))
        residual = x if self.shortcut else self.short(x)
        return self.activation(out + residual)


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_count: int,
        stage_number: int,
        activation: str = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for index in range(block_count):
            blocks.append(
                BasicBlock(
                    in_channels,
                    out_channels,
                    stride=2 if index == 0 and stage_number != 2 else 1,
                    shortcut=index != 0,
                    activation=activation,
                    variant=variant,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class PResNet18(nn.Module):
    """Paddle-style ResNet-18 variant d used by official RT-DETR-R18.

    Outputs the stride-8, stride-16 and stride-32 feature maps with channels
    128, 256 and 512. No pretrained weights are loaded.
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1_1",
                        ConvNormLayer(input_channels, 32, 3, 2, activation="relu"),
                    ),
                    ("conv1_2", ConvNormLayer(32, 32, 3, 1, activation="relu")),
                    ("conv1_3", ConvNormLayer(32, 64, 3, 1, activation="relu")),
                ]
            )
        )
        self.stages = nn.ModuleList(
            [
                ResidualStage(64, 64, 2, 2),
                ResidualStage(64, 128, 2, 3),
                ResidualStage(128, 256, 2, 4),
                ResidualStage(256, 512, 2, 5),
            ]
        )
        self.out_channels = (128, 256, 512)
        self.out_strides = (8, 16, 32)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outputs: List[Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            if stage_index in (1, 2, 3):
                outputs.append(x)
        return outputs[0], outputs[1], outputs[2]


# -----------------------------------------------------------------------------
# Hybrid encoder: AIFI on P5 plus CNN cross-scale feature fusion
# -----------------------------------------------------------------------------


class RepVggBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu") -> None:
        super().__init__()
        self.conv3 = ConvNormLayer(in_channels, out_channels, 3, 1)
        self.conv1 = ConvNormLayer(in_channels, out_channels, 1, 1, padding=0)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv3(x) + self.conv1(x))


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(
            in_channels, hidden_channels, 1, 1, activation=activation
        )
        self.conv2 = ConvNormLayer(
            in_channels, hidden_channels, 1, 1, activation=activation
        )
        self.blocks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, activation=activation)
                for _ in range(num_blocks)
            ]
        )
        self.conv3 = (
            ConvNormLayer(hidden_channels, out_channels, 1, 1, activation=activation)
            if hidden_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(self.blocks(self.conv1(x)) + self.conv2(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = get_activation(activation)

    def forward(self, source: Tensor, position_embedding: Tensor | None = None) -> Tensor:
        query = key = source if position_embedding is None else source + position_embedding
        attended, _ = self.self_attention(query, key, source, need_weights=False)
        source = self.norm1(source + self.dropout1(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(source))))
        return self.norm2(source + self.dropout2(feedforward))


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512),
        feat_strides: Sequence[int] = (8, 16, 32),
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 1,
        expansion: float = 0.5,
        depth_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.feat_strides = tuple(feat_strides)
        self.hidden_dim = hidden_dim
        self.out_channels = (hidden_dim, hidden_dim, hidden_dim)
        self.out_strides = self.feat_strides

        self.input_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
                for channel in in_channels
            ]
        )
        self.aifi_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_dim,
                    nhead,
                    dim_feedforward,
                    dropout=0.0,
                    activation="gelu",
                )
                for _ in range(num_encoder_layers)
            ]
        )

        blocks = max(1, round(3 * depth_mult))
        self.lateral_convs = nn.ModuleList(
            [
                ConvNormLayer(hidden_dim, hidden_dim, 1, 1, activation="silu")
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.fpn_blocks = nn.ModuleList(
            [
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    blocks,
                    expansion,
                    activation="silu",
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.downsample_convs = nn.ModuleList(
            [
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, activation="silu")
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.pan_blocks = nn.ModuleList(
            [
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    blocks,
                    expansion,
                    activation="silu",
                )
                for _ in range(len(in_channels) - 1)
            ]
        )

    @staticmethod
    def build_2d_sincos_position_embedding(
        width: int,
        height: int,
        embed_dim: int = 256,
        temperature: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4")
        grid_w = torch.arange(width, device=device, dtype=dtype)
        grid_h = torch.arange(height, device=device, dtype=dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        position_dim = embed_dim // 4
        omega = torch.arange(position_dim, device=device, dtype=dtype) / position_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[:, None] @ omega[None]
        out_h = grid_h.flatten()[:, None] @ omega[None]
        return torch.cat((out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()), dim=1)[
            None
        ]

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if len(features) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} features, got {len(features)}")
        projected = [
            projection(feature)
            for projection, feature in zip(self.input_projections, features)
        ]

        # AIFI is applied only to the deepest and cheapest P5 feature map.
        height, width = projected[-1].shape[-2:]
        tokens = projected[-1].flatten(2).permute(0, 2, 1)
        position = self.build_2d_sincos_position_embedding(
            width,
            height,
            self.hidden_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for layer in self.aifi_layers:
            tokens = layer(tokens, position)
        projected[-1] = tokens.permute(0, 2, 1).reshape(
            projected[-1].shape[0], self.hidden_dim, height, width
        )

        # Top-down FPN.
        inner_outputs = [projected[-1]]
        feature_count = len(projected)
        for feature_index in range(feature_count - 1, 0, -1):
            high = self.lateral_convs[feature_count - 1 - feature_index](
                inner_outputs[0]
            )
            inner_outputs[0] = high
            upsampled = F.interpolate(high, scale_factor=2.0, mode="nearest")
            low = projected[feature_index - 1]
            fused = self.fpn_blocks[feature_count - 1 - feature_index](
                torch.cat((upsampled, low), dim=1)
            )
            inner_outputs.insert(0, fused)

        # Bottom-up PAN.
        outputs = [inner_outputs[0]]
        for feature_index in range(feature_count - 1):
            downsampled = self.downsample_convs[feature_index](outputs[-1])
            outputs.append(
                self.pan_blocks[feature_index](
                    torch.cat((downsampled, inner_outputs[feature_index + 1]), dim=1)
                )
            )
        return outputs[0], outputs[1], outputs[2]


# -----------------------------------------------------------------------------
# RT-DETR transformer decoder
# -----------------------------------------------------------------------------


def deformable_attention_core(
    value: Tensor,
    spatial_shapes: Sequence[Tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, head_dim = value.shape
    _, query_count, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [height * width for height, width in spatial_shapes]
    value_levels = value.split(split_sizes, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled_levels: List[Tensor] = []

    for level, (height, width) in enumerate(spatial_shapes):
        value_level = (
            value_levels[level]
            .flatten(2)
            .permute(0, 2, 1)
            .reshape(batch_size * num_heads, head_dim, height, width)
        )
        grid_level = (
            sampling_grids[:, :, :, level]
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 1)
        )
        # ``grid_sample`` support for fp16/bfloat16 differs between PyTorch
        # builds. Run this numerical kernel in float32 and cast back so AMP
        # remains safe on the H100 and on older CUDA-enabled PyTorch versions.
        sampled = F.grid_sample(
            value_level.float(),
            grid_level.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).to(value_level.dtype)
        sampled_levels.append(sampled)

    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_heads,
        1,
        query_count,
        num_levels * num_points,
    )
    output = (
        torch.stack(sampled_levels, dim=-2).flatten(-2) * weights
    ).sum(-1)
    output = output.reshape(batch_size, num_heads * head_dim, query_count)
    return output.permute(0, 2, 1)


class MultiScaleDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        total_points = num_heads * num_levels * num_points
        self.sampling_offsets = nn.Linear(embed_dim, total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, total_points)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True).values
        grid = grid.reshape(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(
            1, 1, -1, 1
        )
        self.sampling_offsets.bias.data.copy_((grid * scaling).flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
        value_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, query_count = query.shape[:2]
        value_count = value.shape[1]
        value = self.value_projection(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(
            batch_size,
            value_count,
            self.num_heads,
            self.head_dim,
        )
        offsets = self.sampling_offsets(query).reshape(
            batch_size,
            query_count,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query).reshape(
            batch_size,
            query_count,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        weights = F.softmax(weights, dim=-1).reshape(
            batch_size,
            query_count,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        elif reference_points.shape[-1] == 2:
            normalizer = torch.as_tensor(
                spatial_shapes,
                device=query.device,
                dtype=query.dtype,
            ).flip(-1)
            locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[
                None, None, None, :, None, :
            ]
        else:
            raise ValueError("Reference points must have 2 or 4 coordinates")

        output = deformable_attention_core(value, spatial_shapes, locations, weights)
        return self.output_projection(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_levels: int = 3,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention = MultiScaleDeformableAttention(
            hidden_dim,
            nhead,
            num_levels,
            num_points,
        )
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
        attention_mask: Tensor | None = None,
        query_position: Tensor | None = None,
    ) -> Tensor:
        positioned = target if query_position is None else target + query_position
        attended, _ = self.self_attention(
            positioned,
            positioned,
            target,
            attn_mask=attention_mask,
            need_weights=False,
        )
        target = self.norm1(target + self.dropout1(attended))
        positioned = target if query_position is None else target + query_position
        cross = self.cross_attention(
            positioned,
            reference_points,
            memory,
            spatial_shapes,
        )
        target = self.norm2(target + self.dropout2(cross))
        feedforward = self.linear2(self.dropout3(F.relu(self.linear1(target))))
        target = self.norm3((target + self.dropout4(feedforward)).clamp(-65504, 65504))
        return target


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        eval_index: int = -1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.eval_index = eval_index if eval_index >= 0 else num_layers + eval_index

    def forward(
        self,
        target: Tensor,
        reference_points_unactivated: Tensor,
        memory: Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
        box_heads: nn.ModuleList,
        score_heads: nn.ModuleList,
        query_position_head: nn.Module,
        attention_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        output = target
        output_boxes: List[Tensor] = []
        output_logits: List[Tensor] = []
        reference_detached = reference_points_unactivated.sigmoid()
        previous_reference: Tensor | None = None

        for layer_index, layer in enumerate(self.layers):
            reference_input = reference_detached.unsqueeze(2)
            query_position = query_position_head(reference_detached)
            output = layer(
                output,
                reference_input,
                memory,
                spatial_shapes,
                attention_mask,
                query_position,
            )
            current_box = (
                box_heads[layer_index](output) + inverse_sigmoid(reference_detached)
            ).sigmoid()
            if self.training:
                output_logits.append(score_heads[layer_index](output))
                if layer_index == 0 or previous_reference is None:
                    output_boxes.append(current_box)
                else:
                    output_boxes.append(
                        (
                            box_heads[layer_index](output)
                            + inverse_sigmoid(previous_reference)
                        ).sigmoid()
                    )
            elif layer_index == self.eval_index:
                output_logits.append(score_heads[layer_index](output))
                output_boxes.append(current_box)
                break
            previous_reference = current_box
            reference_detached = current_box.detach() if self.training else current_box

        return torch.stack(output_boxes), torch.stack(output_logits)


# -----------------------------------------------------------------------------
# Contrastive denoising queries used by the default RT-DETR training recipe
# -----------------------------------------------------------------------------


def build_denoising_group(
    targets: Sequence[Dict[str, Tensor]],
    num_classes: int,
    num_queries: int,
    class_embedding: nn.Embedding,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> Tuple[Tensor | None, Tensor | None, Tensor | None, Dict[str, object] | None]:
    if num_denoising <= 0 or not targets:
        return None, None, None, None
    ground_truth_counts = [len(target["labels"]) for target in targets]
    maximum_ground_truth = max(ground_truth_counts, default=0)
    if maximum_ground_truth == 0:
        return None, None, None, None

    device = targets[0]["labels"].device
    batch_size = len(targets)
    group_count = max(1, num_denoising // maximum_ground_truth)
    query_classes = torch.full(
        (batch_size, maximum_ground_truth),
        num_classes,
        dtype=torch.long,
        device=device,
    )
    query_boxes = torch.zeros(batch_size, maximum_ground_truth, 4, device=device)
    padding_mask = torch.zeros(
        batch_size,
        maximum_ground_truth,
        dtype=torch.bool,
        device=device,
    )
    for batch_index, target in enumerate(targets):
        count = len(target["labels"])
        if count:
            query_classes[batch_index, :count] = target["labels"]
            query_boxes[batch_index, :count] = target["boxes"]
            padding_mask[batch_index, :count] = True

    query_classes = query_classes.tile((1, 2 * group_count))
    query_boxes = query_boxes.tile((1, 2 * group_count, 1))
    padding_mask = padding_mask.tile((1, 2 * group_count))

    negative_mask = torch.zeros(
        batch_size,
        maximum_ground_truth * 2,
        1,
        device=device,
    )
    negative_mask[:, maximum_ground_truth:] = 1.0
    negative_mask = negative_mask.tile((1, group_count, 1))
    positive_mask = (1.0 - negative_mask).squeeze(-1) * padding_mask
    positive_indices = torch.nonzero(positive_mask)[:, 1]
    positive_indices = torch.split(
        positive_indices,
        [count * group_count for count in ground_truth_counts],
    )

    actual_denoising_count = maximum_ground_truth * 2 * group_count
    if label_noise_ratio > 0:
        noise_mask = torch.rand_like(query_classes, dtype=torch.float32) < (
            label_noise_ratio * 0.5
        )
        random_labels = torch.randint(
            0,
            num_classes,
            query_classes.shape,
            device=device,
        )
        query_classes = torch.where(
            noise_mask & padding_mask,
            random_labels,
            query_classes,
        )

    if box_noise_scale > 0:
        noisy_xyxy = box_cxcywh_to_xyxy(query_boxes)
        difference = query_boxes[..., 2:].tile((1, 1, 2)) * 0.5 * box_noise_scale
        random_sign = torch.randint_like(query_boxes, 0, 2) * 2.0 - 1.0
        random_part = torch.rand_like(query_boxes)
        random_part = (random_part + 1.0) * negative_mask + random_part * (
            1.0 - negative_mask
        )
        noisy_xyxy = (noisy_xyxy + random_part * random_sign * difference).clamp(0, 1)
        query_boxes = box_xyxy_to_cxcywh(noisy_xyxy)

    query_box_logits = inverse_sigmoid(query_boxes)
    query_class_embeddings = class_embedding(query_classes)
    total_queries = actual_denoising_count + num_queries
    attention_mask = torch.zeros(
        total_queries,
        total_queries,
        dtype=torch.bool,
        device=device,
    )
    attention_mask[actual_denoising_count:, :actual_denoising_count] = True
    group_width = maximum_ground_truth * 2
    for group_index in range(group_count):
        start = group_index * group_width
        end = start + group_width
        attention_mask[start:end, :start] = True
        attention_mask[start:end, end:actual_denoising_count] = True

    metadata: Dict[str, object] = {
        "dn_positive_idx": positive_indices,
        "dn_num_group": group_count,
        "dn_num_split": [actual_denoising_count, num_queries],
    }
    return query_class_embeddings, query_box_logits, attention_mask, metadata


class RTDETRTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: Sequence[int] = (256, 256, 256),
        feat_strides: Sequence[int] = (8, 16, 32),
        num_levels: int = 3,
        num_decoder_points: int = 4,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_index: int = -1,
        auxiliary_loss: bool = True,
        eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.feat_strides = tuple(feat_strides)
        self.num_levels = num_levels
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.auxiliary_loss = auxiliary_loss
        self.eps = eps

        self.input_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
                for channel in feat_channels
            ]
        )
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout=0.0,
            num_levels=num_levels,
            num_points=num_decoder_points,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_decoder_layers,
            eval_index,
        )
        self.denoising_class_embedding = nn.Embedding(
            num_classes + 1,
            hidden_dim,
            padding_idx=num_classes,
        )
        self.query_position_head = MLP(4, hidden_dim * 2, hidden_dim, 2)
        self.encoder_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.encoder_score_head = nn.Linear(hidden_dim, num_classes)
        self.encoder_box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.decoder_score_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)]
        )
        self.decoder_box_heads = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        class_bias = bias_init_with_prob(0.01)
        nn.init.constant_(self.encoder_score_head.bias, class_bias)
        nn.init.zeros_(self.encoder_box_head.layers[-1].weight)
        nn.init.zeros_(self.encoder_box_head.layers[-1].bias)
        for score_head, box_head in zip(
            self.decoder_score_heads,
            self.decoder_box_heads,
        ):
            nn.init.constant_(score_head.bias, class_bias)
            nn.init.zeros_(box_head.layers[-1].weight)
            nn.init.zeros_(box_head.layers[-1].bias)
        nn.init.xavier_uniform_(self.encoder_output[0].weight)
        nn.init.xavier_uniform_(self.query_position_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_position_head.layers[1].weight)

    def _get_encoder_input(
        self,
        features: Sequence[Tensor],
    ) -> Tuple[Tensor, List[Tuple[int, int]]]:
        projected = [
            projection(feature)
            for projection, feature in zip(self.input_projections, features)
        ]
        flattened: List[Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []
        for feature in projected:
            height, width = feature.shape[-2:]
            flattened.append(feature.flatten(2).permute(0, 2, 1))
            spatial_shapes.append((height, width))
        return torch.cat(flattened, dim=1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: Sequence[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
        grid_size: float = 0.05,
    ) -> Tuple[Tensor, Tensor]:
        anchors: List[Tensor] = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack((grid_x, grid_y), dim=-1)
            valid_size = torch.tensor((width, height), device=device, dtype=dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_size
            width_height = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(
                torch.cat((grid_xy, width_height), dim=-1).reshape(1, -1, 4)
            )
        anchors_tensor = torch.cat(anchors, dim=1)
        valid_mask = (
            (anchors_tensor > self.eps) & (anchors_tensor < 1.0 - self.eps)
        ).all(dim=-1, keepdim=True)
        anchors_unactivated = inverse_sigmoid(anchors_tensor)
        anchors_unactivated = torch.where(
            valid_mask,
            anchors_unactivated,
            torch.full_like(anchors_unactivated, float("inf")),
        )
        return anchors_unactivated, valid_mask

    def _get_decoder_input(
        self,
        memory: Tensor,
        spatial_shapes: Sequence[Tuple[int, int]],
        denoising_classes: Tensor | None,
        denoising_boxes: Tensor | None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = memory.shape[0]
        anchors, valid_mask = self._generate_anchors(
            spatial_shapes,
            memory.device,
            memory.dtype,
        )
        memory = memory * valid_mask.to(memory.dtype)
        output_memory = self.encoder_output(memory)
        encoder_logits = self.encoder_score_head(output_memory)
        encoder_boxes_unactivated = self.encoder_box_head(output_memory) + anchors
        top_indices = encoder_logits.max(dim=-1).values.topk(
            self.num_queries,
            dim=1,
        ).indices
        reference_unactivated = encoder_boxes_unactivated.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        encoder_top_boxes = reference_unactivated.sigmoid()
        encoder_top_logits = encoder_logits.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.num_classes),
        )
        target = output_memory.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        ).detach()
        if denoising_boxes is not None:
            reference_unactivated = torch.cat(
                (denoising_boxes, reference_unactivated),
                dim=1,
            )
        if denoising_classes is not None:
            target = torch.cat((denoising_classes, target), dim=1)
        return target, reference_unactivated.detach(), encoder_top_boxes, encoder_top_logits

    @staticmethod
    def _set_auxiliary_outputs(
        logits: Sequence[Tensor],
        boxes: Sequence[Tensor],
    ) -> List[Dict[str, Tensor]]:
        return [
            {"pred_logits": layer_logits, "pred_boxes": layer_boxes}
            for layer_logits, layer_boxes in zip(logits, boxes)
        ]

    def forward(
        self,
        features: Sequence[Tensor],
        targets: Sequence[Dict[str, Tensor]] | None = None,
    ) -> Dict[str, Tensor | List[Dict[str, Tensor]] | Dict[str, object]]:
        memory, spatial_shapes = self._get_encoder_input(features)
        if self.training and targets is not None and self.num_denoising > 0:
            denoising_classes, denoising_boxes, attention_mask, dn_metadata = (
                build_denoising_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embedding,
                    self.num_denoising,
                    self.label_noise_ratio,
                    self.box_noise_scale,
                )
            )
        else:
            denoising_classes = None
            denoising_boxes = None
            attention_mask = None
            dn_metadata = None

        target, reference_unactivated, encoder_boxes, encoder_logits = (
            self._get_decoder_input(
                memory,
                spatial_shapes,
                denoising_classes,
                denoising_boxes,
            )
        )
        decoder_boxes, decoder_logits = self.decoder(
            target,
            reference_unactivated,
            memory,
            spatial_shapes,
            self.decoder_box_heads,
            self.decoder_score_heads,
            self.query_position_head,
            attention_mask,
        )

        if self.training and dn_metadata is not None:
            denoising_count, query_count = dn_metadata["dn_num_split"]  # type: ignore[index]
            dn_boxes, decoder_boxes = torch.split(
                decoder_boxes,
                [int(denoising_count), int(query_count)],
                dim=2,
            )
            dn_logits, decoder_logits = torch.split(
                decoder_logits,
                [int(denoising_count), int(query_count)],
                dim=2,
            )
        else:
            dn_boxes = None
            dn_logits = None

        output: Dict[str, object] = {
            "pred_logits": decoder_logits[-1],
            "pred_boxes": decoder_boxes[-1],
        }
        if self.training and self.auxiliary_loss:
            auxiliary = self._set_auxiliary_outputs(
                decoder_logits[:-1],
                decoder_boxes[:-1],
            )
            auxiliary.extend(
                self._set_auxiliary_outputs([encoder_logits], [encoder_boxes])
            )
            output["aux_outputs"] = auxiliary
            if dn_metadata is not None and dn_logits is not None and dn_boxes is not None:
                output["dn_aux_outputs"] = self._set_auxiliary_outputs(
                    dn_logits,
                    dn_boxes,
                )
                output["dn_meta"] = dn_metadata
        return output  # type: ignore[return-value]


class RTDETR(nn.Module):
    """RT-DETR-R18 assembled from the official backbone/encoder/decoder design."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 20,
        num_queries: int = 300,
        hidden_dim: int = 256,
        num_decoder_layers: int = 3,
        num_denoising: int = 100,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.backbone = PResNet18(input_channels)
        self.encoder = HybridEncoder(
            in_channels=self.backbone.out_channels,
            feat_strides=self.backbone.out_strides,
            hidden_dim=hidden_dim,
            expansion=0.5,
        )
        self.decoder = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=self.encoder.out_channels,
            feat_strides=self.encoder.out_strides,
            num_decoder_layers=num_decoder_layers,
            num_denoising=num_denoising,
        )

    def forward(
        self,
        images: Tensor,
        targets: Sequence[Dict[str, Tensor]] | None = None,
    ) -> Dict[str, Tensor | List[Dict[str, Tensor]] | Dict[str, object]]:
        features = self.backbone(images)
        encoded = self.encoder(features)
        return self.decoder(encoded, targets)

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        max_detections: int | None = None,
    ) -> List[Tensor]:
        """Return NMS-free detections as [x1,y1,x2,y2,score,class].

        Coordinates are normalized to 0..1. Validation maps them into image
        pixels. One best class is retained for each transformer query.
        """
        was_training = self.training
        self.eval()
        output = self(images)
        logits = output["pred_logits"]
        boxes = output["pred_boxes"]
        if not isinstance(logits, Tensor) or not isinstance(boxes, Tensor):
            raise TypeError("RT-DETR output tensors are missing")
        scores, classes = logits.sigmoid().max(dim=-1)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        result: List[Tensor] = []
        for image_index in range(images.shape[0]):
            order = scores[image_index].argsort(descending=True)
            if max_detections is not None:
                order = order[:max_detections]
            result.append(
                torch.cat(
                    (
                        boxes_xyxy[image_index, order],
                        scores[image_index, order, None],
                        classes[image_index, order, None].to(boxes.dtype),
                    ),
                    dim=1,
                )
            )
        if was_training:
            self.train()
        return result


if __name__ == "__main__":
    model = RTDETR(num_classes=20, num_denoising=0)
    model.eval()
    sample = torch.randn(1, 3, 416, 416)
    with torch.no_grad():
        output = model(sample)
    print("pred_logits:", tuple(output["pred_logits"].shape))
    print("pred_boxes :", tuple(output["pred_boxes"].shape))
    print("parameters :", f"{sum(p.numel() for p in model.parameters()):,}")
