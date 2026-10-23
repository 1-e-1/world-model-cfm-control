"""Token-conditioned second-order point-flow model.

The state velocity is point motion in reconstruction flow space. It is unrelated to robot
joint velocity. Point-to-point self-attention is intentionally avoided so compute scales as
O(N*K_context) rather than O(N^2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from codex_zbot.config import CFMConfig
from codex_zbot.utils.metrics import flow_statistics


@dataclass
class CFMCondition:
    image: torch.Tensor
    pose: torch.Tensor
    observed_points: torch.Tensor
    observed_mask: torch.Tensor
    previous_points: torch.Tensor
    previous_velocity: torch.Tensor
    confidence: torch.Tensor
    particle_type: torch.Tensor
    semantic: torch.Tensor | None = None


@dataclass
class FlowState:
    position: torch.Tensor
    velocity: torch.Tensor
    confidence: torch.Tensor
    acceleration: torch.Tensor


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, query_count, _ = query.shape
        key_count = key_value.shape[1]
        q = self.q(query).view(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(batch, key_count, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).view(batch, key_count, self.heads, self.head_dim).transpose(1, 2)
        attention_mask = None
        if key_mask is not None:
            attention_mask = key_mask[:, None, None, :].to(torch.bool)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.out(output.transpose(1, 2).reshape(batch, query_count, self.dim))


class Tokenizer(nn.Module):
    def __init__(self, dim: int, token_count: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, token_count, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.attention = CrossAttention(dim, heads)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, source: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query.expand(source.shape[0], -1, -1)
        tokens = query + self.attention(self.query_norm(query), self.source_norm(source), mask)
        return self.output_norm(tokens)


class ImageEncoder(nn.Module):
    def __init__(self, channels: int, dim: int, tokens: int, heads: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 48, 5, stride=2, padding=2),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
        )
        self.projection = nn.Linear(192, dim)
        self.tokenizer = Tokenizer(dim, tokens, heads)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.conv(image).flatten(2).transpose(1, 2)
        return self.tokenizer(self.projection(feature))


class PointEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int, tokens: int, heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, dim),
        )
        self.tokenizer = Tokenizer(dim, tokens, heads)

    def forward(self, points: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.tokenizer(self.mlp(points), mask)


class VectorTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int, tokens: int):
        super().__init__()
        self.tokens = tokens
        self.network = nn.Sequential(nn.Linear(input_dim, dim), nn.SiLU(), nn.Linear(dim, dim * tokens))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).reshape(value.shape[0], self.tokens, -1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 96):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=time.device, dtype=time.dtype) / max(1, half - 1)
        )
        phase = time[:, None] * frequencies[None]
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class ConditionedPointBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = CrossAttention(dim, heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Linear(ffn_dim, dim))

    def forward(self, points: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        points = points + self.attention(self.query_norm(points), self.context_norm(context))
        return points + self.ffn(self.ffn_norm(points))


class AccelerationField(nn.Module):
    def __init__(self, cfg: CFMConfig):
        super().__init__()
        self.time_embedding = TimeEmbedding(96)
        # xyz + velocity + confidence + particle_type + time embedding
        self.input_projection = nn.Linear(3 + 3 + 1 + 1 + 96, cfg.dim)
        self.blocks = nn.ModuleList(
            [ConditionedPointBlock(cfg.dim, cfg.heads, cfg.ffn_dim) for _ in range(cfg.blocks)]
        )
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.acceleration = nn.Linear(cfg.dim, 3)
        self.confidence = nn.Linear(cfg.dim, 1)

    def forward(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        confidence: torch.Tensor,
        particle_type: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_feature = self.time_embedding(time)[:, None].expand(-1, position.shape[1], -1)
        feature = self.input_projection(
            torch.cat([position, velocity, confidence, particle_type, time_feature], dim=-1)
        )
        for block in self.blocks:
            feature = block(feature, context)
        feature = self.output_norm(feature)
        return self.acceleration(feature), self.confidence(feature)


class SecondOrderCFM(nn.Module):
    def __init__(self, cfg: CFMConfig):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg.image_channels, cfg.dim, cfg.image_tokens, cfg.heads)
        self.previous_encoder = PointEncoder(3 + 3 + 1 + 1, cfg.dim, cfg.point_tokens, cfg.heads)
        self.observed_encoder = PointEncoder(3 + 1, cfg.dim, cfg.point_tokens, cfg.heads)
        self.pose_encoder = VectorTokenEncoder(7, cfg.dim, cfg.pose_tokens)
        self.state_encoder = VectorTokenEncoder(6, cfg.dim, cfg.state_tokens)
        self.semantic_embedding = (
            nn.Embedding(cfg.semantic_classes, cfg.dim) if cfg.semantic_classes > 0 else None
        )
        self.semantic_tokenizer = Tokenizer(cfg.dim, 8, cfg.heads) if cfg.semantic_classes > 0 else None
        self.field = AccelerationField(cfg)

    def encode_condition(self, condition: CFMCondition) -> torch.Tensor:
        previous_feature = torch.cat(
            [
                condition.previous_points,
                condition.previous_velocity,
                condition.confidence,
                condition.particle_type,
            ],
            dim=-1,
        )
        observed_feature = torch.cat(
            [condition.observed_points, condition.observed_mask.unsqueeze(-1).to(condition.observed_points.dtype)],
            dim=-1,
        )
        zeros = torch.zeros_like(condition.previous_points)
        flow_stats = flow_statistics(
            condition.previous_points, condition.previous_points, condition.previous_velocity, zeros
        )
        tokens = [
            self.image_encoder(condition.image),
            self.previous_encoder(previous_feature),
            self.observed_encoder(observed_feature, condition.observed_mask),
            self.pose_encoder(condition.pose),
            self.state_encoder(flow_stats),
        ]
        if self.semantic_embedding is not None and condition.semantic is not None:
            semantic = condition.semantic.long().flatten(1).clamp(0, self.cfg.semantic_classes - 1)
            tokens.append(self.semantic_tokenizer(self.semantic_embedding(semantic)))
        return torch.cat(tokens, dim=1)

    def acceleration(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        confidence: torch.Tensor,
        particle_type: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.field(position, velocity, confidence, particle_type, time, context)

    def integrate(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        confidence: torch.Tensor,
        particle_type: torch.Tensor,
        context: torch.Tensor,
        num_steps: int | None = None,
        solver: str | None = None,
        vmax: float | None = None,
    ) -> FlowState:
        num_steps = num_steps or self.cfg.ode_steps
        solver = solver or self.cfg.solver
        vmax = vmax or self.cfg.vmax
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        batch = position.shape[0]
        dt = 1.0 / num_steps
        acceleration = torch.zeros_like(position)
        confidence_logit = torch.logit(confidence.clamp(1.0e-4, 1.0 - 1.0e-4))
        for step in range(num_steps):
            time = torch.full((batch,), step * dt, device=position.device, dtype=position.dtype)
            acceleration_1, confidence_logit = self.field(
                position, velocity, confidence, particle_type, time, context
            )
            if self.cfg.dynamics_order == 1:
                previous_velocity = velocity
                velocity = acceleration_1.clamp(-vmax, vmax)
                position = position + dt * velocity
                acceleration = (velocity - previous_velocity) / dt
                confidence = torch.sigmoid(confidence_logit)
                continue
            if self.cfg.dynamics_order != 2:
                raise ValueError(f"dynamics_order must be 1 or 2, got {self.cfg.dynamics_order}")
            if solver == "midpoint":
                middle_position = position + 0.5 * dt * velocity
                middle_velocity = velocity + 0.5 * dt * acceleration_1
                middle_time = time + 0.5 * dt
                acceleration, confidence_logit = self.field(
                    middle_position, middle_velocity, confidence, particle_type, middle_time, context
                )
            elif solver == "euler":
                acceleration = acceleration_1
            else:
                raise ValueError(f"Unknown ODE solver: {solver}")
            velocity = (velocity + dt * acceleration).clamp(-vmax, vmax)
            position = position + dt * velocity
            confidence = torch.sigmoid(confidence_logit)
        final_time = torch.ones(batch, device=position.device, dtype=position.dtype)
        terminal_field, confidence_logit = self.field(
            position, velocity, confidence, particle_type, final_time, context
        )
        acceleration = terminal_field if self.cfg.dynamics_order == 2 else torch.zeros_like(terminal_field)
        if self.cfg.dynamics_order == 1:
            velocity = terminal_field.clamp(-vmax, vmax)
        return FlowState(position, velocity, torch.sigmoid(confidence_logit), acceleration)

    def forward(self, condition: CFMCondition, num_steps: int | None = None) -> FlowState:
        context = self.encode_condition(condition)
        return self.integrate(
            condition.previous_points,
            condition.previous_velocity,
            condition.confidence,
            condition.particle_type,
            context,
            num_steps=num_steps,
        )
