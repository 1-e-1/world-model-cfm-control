"""Rolling RGB-D window used by online V4 reconstruction."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import OccupancyDriftConfig


class OnlineV4ObservationBuffer:
    """Keep the exact temporal span used during V4 training.

    Reset left-padding repeats the first observation.  Once enough frames have
    arrived, :meth:`batch` returns the configured number of distinct
    observations spaced by the training stride.
    """

    def __init__(self, num_envs: int, cfg: OccupancyDriftConfig, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.span = (cfg.sequence_length - 1) * cfg.frame_stride + 1
        size = cfg.image_size
        self.rgb = torch.zeros(num_envs, self.span, 3, size, size, device=self.device)
        self.depth = torch.zeros(num_envs, self.span, 1, size, size, device=self.device)
        self.pose = torch.zeros(num_envs, self.span, 7, device=self.device)
        self.intrinsics = torch.eye(3, device=self.device).repeat(num_envs, 1, 1)
        self.valid_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def _prepare(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
            raise ValueError("rgb must have shape (B,H,W,3/4)")
        if depth.ndim == 3:
            depth = depth[..., None]
        if depth.ndim != 4 or depth.shape[-1] != 1:
            raise ValueError("depth must have shape (B,H,W) or (B,H,W,1)")
        source_height, source_width = rgb.shape[1:3]
        rgb_chw = rgb[..., :3].permute(0, 3, 1, 2).float()
        if rgb_chw.max() > 1.5:
            rgb_chw = rgb_chw / 255.0
        depth_chw = depth.permute(0, 3, 1, 2).float()
        target = (self.cfg.image_size, self.cfg.image_size)
        rgb_chw = F.interpolate(rgb_chw, target, mode="bilinear", align_corners=False)
        depth_chw = F.interpolate(depth_chw, target, mode="nearest")
        scaled = intrinsics.float().clone()
        scaled[:, 0, [0, 2]] *= self.cfg.image_size / source_width
        scaled[:, 1, [1, 2]] *= self.cfg.image_size / source_height
        return rgb_chw, depth_chw, scaled

    def reset(
        self,
        env_ids: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> None:
        rgb, depth, intrinsics = self._prepare(rgb, depth, intrinsics)
        self.rgb[env_ids] = rgb[:, None]
        self.depth[env_ids] = depth[:, None]
        self.pose[env_ids] = pose[:, None]
        self.intrinsics[env_ids] = intrinsics
        self.valid_count[env_ids] = 1

    def append(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> None:
        rgb, depth, intrinsics = self._prepare(rgb, depth, intrinsics)
        self.rgb = torch.roll(self.rgb, shifts=-1, dims=1)
        self.depth = torch.roll(self.depth, shifts=-1, dims=1)
        self.pose = torch.roll(self.pose, shifts=-1, dims=1)
        self.rgb[:, -1] = rgb
        self.depth[:, -1] = depth
        self.pose[:, -1] = pose
        self.intrinsics = intrinsics
        self.valid_count.clamp_max_(self.span - 1).add_(1)

    def batch(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        selected = torch.arange(0, self.span, self.cfg.frame_stride, device=self.device)
        if len(selected) != self.cfg.sequence_length:
            raise RuntimeError("temporal buffer selection disagrees with drift configuration")
        index = slice(None) if env_ids is None else env_ids
        return {
            "rgb": self.rgb[index][:, selected],
            "depth": self.depth[index][:, selected],
            "pose": self.pose[index][:, selected],
            "intrinsics": self.intrinsics[index],
        }
