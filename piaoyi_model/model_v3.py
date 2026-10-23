from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OccupancyDriftConfig
from .evidence import RayEvidence, evidence_grid, fixed_grid_centers
from .attention import CrossAttention


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SpatioTemporalRGBDEncoder(nn.Module):
    """Lift image features into world coordinates before temporal fusion."""

    def __init__(self, cfg: OccupancyDriftConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(5, 48, 5, 2, 2), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
        )
        self.visual = nn.Linear(128, cfg.dim)
        self.geometry = nn.Sequential(nn.Linear(11, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.frame_token = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        self.temporal_position = nn.Parameter(torch.randn(1, cfg.sequence_length, cfg.dim) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.ffn_dim, batch_first=True, norm_first=True, activation="gelu"
        )
        temporal_layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.ffn_dim, batch_first=True, norm_first=True, activation="gelu"
        )
        self.spatial = (
            nn.TransformerEncoder(spatial_layer, cfg.spatial_transformer_layers)
            if cfg.spatial_transformer_layers > 0 else None
        )
        self.temporal = (
            nn.TransformerEncoder(temporal_layer, cfg.temporal_transformer_layers)
            if cfg.temporal_transformer_layers > 0 else None
        )
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.register_buffer("lower", torch.tensor(cfg.workspace_min).view(1, 1, 1, 3))
        self.register_buffer("upper", torch.tensor(cfg.workspace_max).view(1, 1, 1, 3))

    def forward(
        self,
        images: torch.Tensor,
        pose: torch.Tensor,
        evidence: RayEvidence,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = images.shape
        feature = self.conv(images.reshape(batch * frames, channels, height, width))
        feature = F.interpolate(feature, (self.cfg.ray_grid_size, self.cfg.ray_grid_size), mode="bilinear", align_corners=False)
        # Parameter-free feature pyramid: retain local edges while injecting a
        # wider receptive field.  It improves partially observed object shape
        # without invalidating existing checkpoints.
        feature = feature + 0.5 * F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        visual = self.visual(feature.flatten(2).transpose(1, 2)).reshape(batch, frames, -1, self.cfg.dim)
        xyz = 2.0 * (evidence.surface_points - self.lower) / (self.upper - self.lower).clamp_min(1.0e-6) - 1.0
        time = torch.linspace(-1, 1, frames, device=images.device, dtype=images.dtype)
        time = time.view(1, frames, 1, 1).expand(batch, -1, xyz.shape[2], -1)
        pose_expanded = pose[:, :, None, :].expand(-1, -1, xyz.shape[2], -1)
        patch = visual + self.geometry(torch.cat([xyz, time, pose_expanded], dim=-1))

        patch = patch.reshape(batch * frames, -1, self.cfg.dim)
        patch_mask = evidence.surface_mask.reshape(batch * frames, -1)
        frame_token = self.frame_token.expand(batch * frames, -1, -1)
        spatial_input = torch.cat([frame_token, patch], dim=1)
        spatial_mask = torch.cat(
            [torch.zeros(batch * frames, 1, device=images.device, dtype=torch.bool), ~patch_mask], dim=1
        )
        spatial = (
            self.spatial(spatial_input, src_key_padding_mask=spatial_mask)
            if self.spatial is not None else spatial_input
        )
        frame_summary = spatial[:, 0].reshape(batch, frames, self.cfg.dim)
        frame_summary = frame_summary + self.temporal_position[:, :frames]
        frame_valid = evidence.surface_mask.any(dim=-1)
        # A view containing only background/ground must not become temporal
        # evidence.  Keep one token unmasked only for all-invalid numerical
        # fallback; normal collection guarantees informative views.
        all_invalid = ~frame_valid.any(dim=1)
        if all_invalid.any():
            frame_valid = frame_valid.clone()
            frame_valid[all_invalid, 0] = True
        quality = evidence.surface_mask.float().mean(dim=-1, keepdim=True)
        quality = quality / quality.amax(dim=1, keepdim=True).clamp_min(1.0e-6)
        frame_summary = frame_summary * (0.5 + 0.5 * quality)
        if self.temporal is not None:
            frame_summary = self.temporal(frame_summary, src_key_padding_mask=~frame_valid)
        patch = spatial[:, 1:].reshape(batch, frames, -1, self.cfg.dim) + frame_summary[:, :, None]
        patch = self.output_norm(patch).flatten(1, 2)
        valid = evidence.surface_mask.flatten(1, 2)
        # Cross-attention must always have at least one valid key.
        valid[:, 0] = True
        return patch, valid, frame_summary


class GridMemoryBlock(nn.Module):
    def __init__(self, cfg: OccupancyDriftConfig):
        super().__init__()
        self.resolution = cfg.memory_resolution
        self.query_norm = nn.LayerNorm(cfg.dim)
        self.source_norm = nn.LayerNorm(cfg.dim)
        self.cross = CrossAttention(cfg.dim, cfg.heads)
        self.conv_norm = nn.GroupNorm(8, cfg.dim)
        self.conv = nn.Sequential(
            nn.Conv3d(cfg.dim, cfg.dim, 3, padding=1, groups=cfg.dim),
            nn.SiLU(),
            nn.Conv3d(cfg.dim, cfg.dim, 1),
        )
        self.ffn = FeedForward(cfg.dim, cfg.ffn_dim)

    def forward(self, memory: torch.Tensor, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        memory = memory + self.cross(self.query_norm(memory), self.source_norm(source), source_mask)
        batch, cells, dim = memory.shape
        grid = memory.reshape(batch, self.resolution, self.resolution, self.resolution, dim).permute(0, 4, 1, 2, 3)
        grid = grid + self.conv(self.conv_norm(grid))
        memory = grid.permute(0, 2, 3, 4, 1).reshape(batch, cells, dim)
        return self.ffn(memory)


@dataclass
class V3Memory:
    tokens: torch.Tensor
    tri_state_logits: torch.Tensor
    frame_tokens: torch.Tensor
    evidence_values: torch.Tensor
    surface_cells: torch.Tensor
    free_cells: torch.Tensor


class GeometryMemory(nn.Module):
    def __init__(self, cfg: OccupancyDriftConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpatioTemporalRGBDEncoder(cfg)
        self.position = nn.Sequential(nn.Linear(3, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.evidence = nn.Sequential(nn.Linear(2, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.blocks = nn.ModuleList([GridMemoryBlock(cfg) for _ in range(cfg.memory_cross_attention_layers)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.tri_state_head = nn.Linear(cfg.dim, 3)

    def forward(self, images: torch.Tensor, pose: torch.Tensor, rays: RayEvidence) -> V3Memory:
        source, source_mask, frame_tokens = self.encoder(images, pose, rays)
        grid_values, surface_cells, free_cells = evidence_grid(rays, self.cfg)
        centers = fixed_grid_centers(self.cfg, images.device, images.dtype)
        lower = torch.as_tensor(self.cfg.workspace_min, device=images.device, dtype=images.dtype)
        upper = torch.as_tensor(self.cfg.workspace_max, device=images.device, dtype=images.dtype)
        normalized = 2.0 * (centers - lower) / (upper - lower).clamp_min(1.0e-6) - 1.0
        memory = self.position(normalized)[None].expand(images.shape[0], -1, -1) + self.evidence(grid_values)
        for block in self.blocks:
            memory = block(memory, source, source_mask)
        memory = self.norm(memory)
        return V3Memory(memory, self.tri_state_head(memory), frame_tokens, grid_values, surface_cells, free_cells)


class LatentPool(nn.Module):
    def __init__(self, dim: int, heads: int, count: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, count, dim) * 0.02)
        self.qnorm = nn.LayerNorm(dim)
        self.snorm = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, heads)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(source.shape[0], -1, -1)
        return query + self.cross(self.qnorm(query), self.snorm(source))


class SetConditionedBlock(nn.Module):
    """Low-rank particle interaction plus cross-attention to scene memory."""

    def __init__(self, cfg: OccupancyDriftConfig):
        super().__init__()
        self.point_norm = nn.LayerNorm(cfg.dim)
        self.context_norm = nn.LayerNorm(cfg.dim)
        self.context_cross = CrossAttention(cfg.dim, cfg.heads)
        self.latent = nn.Parameter(torch.randn(1, cfg.set_latent_tokens, cfg.dim) * 0.02)
        self.to_latent = CrossAttention(cfg.dim, cfg.heads)
        self.to_point = CrossAttention(cfg.dim, cfg.heads)
        self.latent_norm = nn.LayerNorm(cfg.dim)
        self.ffn = FeedForward(cfg.dim, cfg.ffn_dim)

    def forward(self, points: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        points = points + self.context_cross(self.point_norm(points), self.context_norm(context))
        latent = self.latent.expand(points.shape[0], -1, -1)
        latent = latent + self.to_latent(self.latent_norm(latent), self.point_norm(points))
        points = points + self.to_point(self.point_norm(points), self.latent_norm(latent))
        return self.ffn(points)


@dataclass
class OccupancyDriftV3Output:
    points: torch.Tensor
    memory: V3Memory
    coarse_cell_ids: torch.Tensor


class OccupancyDriftGeneratorV3(nn.Module):
    """Coarse-to-fine one-NFE generator conditioned on a 3D temporal memory."""

    def __init__(self, cfg: OccupancyDriftConfig):
        super().__init__()
        if cfg.architecture != "v3_spatiotemporal":
            raise ValueError("V3 model requires architecture='v3_spatiotemporal'")
        self.cfg = cfg
        self.geometry_memory = GeometryMemory(cfg)
        self.memory_pool = LatentPool(cfg.dim, cfg.heads, cfg.context_tokens)
        self.noise_input = nn.Sequential(nn.Linear(7, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.cell_feature = nn.Linear(cfg.dim, cfg.dim)
        self.blocks = nn.ModuleList([SetConditionedBlock(cfg) for _ in range(cfg.blocks)])
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.local_displacement = nn.Linear(cfg.dim, 3)
        self.global_offset = nn.Linear(cfg.dim, 3)
        nn.init.zeros_(self.local_displacement.weight)
        nn.init.zeros_(self.local_displacement.bias)
        nn.init.zeros_(self.global_offset.weight)
        nn.init.zeros_(self.global_offset.bias)
        self.field_head = nn.Sequential(
            nn.Linear(cfg.dim + 3, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim // 2), nn.SiLU(), nn.Linear(cfg.dim // 2, 2)
        )
        self.register_buffer("lower", torch.tensor(cfg.workspace_min).view(1, 1, 3))
        self.register_buffer("upper", torch.tensor(cfg.workspace_max).view(1, 1, 3))
        self.register_buffer("cell_centers", fixed_grid_centers(cfg, torch.device("cpu"), torch.float32))

    def encode_memory(self, images: torch.Tensor, pose: torch.Tensor, evidence: RayEvidence) -> V3Memory:
        return self.geometry_memory(images, pose, evidence)

    def _coarse_ids(self, logits: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        occupied = logits.softmax(dim=-1)[..., 0]
        probabilities = (occupied + 1.0e-5).pow(1.0 / self.cfg.coarse_temperature)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        cdf = probabilities.cumsum(dim=-1).contiguous()
        uniform = 0.5 * (1.0 + torch.erf(noise[..., 0] / (2.0**0.5)))
        ids = torch.searchsorted(cdf, uniform.flatten(1), right=False)
        return ids.clamp_max(cdf.shape[-1] - 1).reshape(uniform.shape)

    def generate(
        self,
        noise: torch.Tensor,
        memory: V3Memory,
        observed_points: torch.Tensor,
        observed_mask: torch.Tensor,
        teacher_cell_ids: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, samples, point_count, _ = noise.shape
        cell_ids = self._coarse_ids(memory.tri_state_logits, noise)
        if teacher_cell_ids is not None and teacher_forcing_ratio > 0:
            forcing_uniform = 0.5 * (1.0 + torch.erf((noise[..., 0] + noise[..., 1]) / 2.0))
            cell_ids = torch.where(forcing_uniform < teacher_forcing_ratio, teacher_cell_ids, cell_ids)
        centers = self.cell_centers.to(noise).view(1, 1, -1, 3).expand(batch, samples, -1, -1)
        selected_centers = torch.gather(centers, 2, cell_ids[..., None].expand(-1, -1, -1, 3))
        memory_expanded = memory.tokens[:, None].expand(-1, samples, -1, -1)
        selected_feature = torch.gather(
            memory_expanded, 2, cell_ids[..., None].expand(-1, -1, -1, self.cfg.dim)
        )

        valid_count = observed_mask.sum(dim=1).clamp_min(1)
        anchor_uniform = 0.5 * (1.0 + torch.erf(noise[..., 1] / (2.0**0.5)))
        anchor_ids = (anchor_uniform * valid_count[:, None, None].to(noise.dtype)).long()
        anchor_ids = torch.minimum(anchor_ids, (valid_count - 1)[:, None, None])
        observed_expanded = observed_points[:, None].expand(-1, samples, -1, -1)
        anchors = torch.gather(observed_expanded, 2, anchor_ids[..., None].expand(-1, -1, -1, 3))
        local_uniform = 0.5 * (1.0 + torch.erf(noise[..., 2:3] / (2.0**0.5)))
        has_observation = observed_mask.any(dim=1)[:, None, None, None]
        local_mask = (local_uniform >= self.cfg.global_particle_ratio) & has_observation
        base = torch.where(local_mask, anchors, selected_centers)
        lower = self.lower.view(1, 1, 1, 3)
        upper = self.upper.view(1, 1, 1, 3)
        base_normalized = 2.0 * (base - lower) / (upper - lower).clamp_min(1.0e-6) - 1.0
        feature = self.noise_input(torch.cat([noise, base_normalized, local_mask.to(noise.dtype)], dim=-1))
        feature = feature + self.cell_feature(selected_feature)
        feature = feature.reshape(batch * samples, point_count, self.cfg.dim)
        context = torch.cat([self.memory_pool(memory.tokens), memory.frame_tokens], dim=1)
        context = context[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, -1, self.cfg.dim)
        for block in self.blocks:
            feature = block(feature, context)
        feature = self.output_norm(feature)
        noise_flat = noise.reshape(batch * samples, point_count, 3)
        local = anchors.reshape(batch * samples, point_count, 3) + self.cfg.max_displacement_m * torch.tanh(
            self.local_displacement(feature) + 0.1 * noise_flat
        )
        cell_size = (self.upper - self.lower) / self.cfg.memory_resolution
        global_points = selected_centers.reshape(batch * samples, point_count, 3) + cell_size * self.cfg.coarse_cell_expansion * torch.tanh(
            self.global_offset(feature) + 0.1 * noise_flat
        )
        points = torch.where(local_mask.reshape(batch * samples, point_count, 1), local, global_points)
        points = torch.maximum(torch.minimum(points, self.upper), self.lower)
        return points.reshape(batch, samples, point_count, 3), cell_ids

    def decode_field(self, memory: V3Memory, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode occupancy logit and truncated SDF at arbitrary world queries."""
        resolution = self.cfg.memory_resolution
        batch, count, _ = query.shape
        normalized = 2.0 * (query - self.lower) / (self.upper - self.lower).clamp_min(1.0e-6) - 1.0
        # Manual trilinear lookup is deterministic on CUDA, unlike the
        # backward pass of grid_sample for 5-D tensors.
        coordinate = ((normalized + 1.0) * 0.5 * (resolution - 1)).clamp(0, resolution - 1)
        base = coordinate.floor().long()
        fraction = coordinate - base.to(coordinate.dtype)
        sampled = torch.zeros(batch, count, self.cfg.dim, device=query.device, dtype=memory.tokens.dtype)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    xyz = base + torch.tensor([dx, dy, dz], device=query.device)
                    xyz = xyz.clamp(0, resolution - 1)
                    index = xyz[..., 0] * resolution * resolution + xyz[..., 1] * resolution + xyz[..., 2]
                    corner = torch.gather(memory.tokens, 1, index[..., None].expand(-1, -1, self.cfg.dim))
                    wx = fraction[..., 0] if dx else 1.0 - fraction[..., 0]
                    wy = fraction[..., 1] if dy else 1.0 - fraction[..., 1]
                    wz = fraction[..., 2] if dz else 1.0 - fraction[..., 2]
                    sampled = sampled + corner * (wx * wy * wz)[..., None]
        output = self.field_head(torch.cat([sampled, normalized], dim=-1))
        occupancy_logit = output[..., 0]
        sdf = self.cfg.sdf_truncation_m * torch.tanh(output[..., 1])
        return occupancy_logit, sdf

    def forward(
        self,
        noise: torch.Tensor,
        images: torch.Tensor,
        pose: torch.Tensor,
        evidence: RayEvidence,
        observed_points: torch.Tensor,
        observed_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | OccupancyDriftV3Output:
        memory = self.encode_memory(images, pose, evidence)
        points, cell_ids = self.generate(noise, memory, observed_points, observed_mask)
        if return_aux:
            return OccupancyDriftV3Output(points, memory, cell_ids)
        return points
