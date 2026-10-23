"""Sequence-preserving loader for collected IsaacLab episodes."""
from __future__ import annotations

import glob
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def split_episode_paths(
    data_dir: str | Path,
    split: Literal["train", "val", "test"],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[Path]:
    paths = [Path(p) for p in sorted(glob.glob(str(Path(data_dir) / "ep_*.npz")))]
    if not paths:
        raise FileNotFoundError(f"No ep_*.npz files found in {data_dir}")
    if abs(sum(ratios) - 1.0) > 1.0e-6:
        raise ValueError("split ratios must sum to one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    # A freshly collected on-policy replay directory may legitimately contain
    # only one episode.  It still belongs to the training split.
    n_train = max(1, int(len(paths) * ratios[0]))
    n_val = min(int(len(paths) * ratios[1]), len(paths) - n_train)
    ranges = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    selected = ranges[split]
    if len(selected) == 0:
        raise RuntimeError(f"The {split} split is empty; collect more episodes or change ratios")
    return [paths[int(i)] for i in selected]


def _sample_gt(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic approximate FPS on a bounded candidate set."""
    points = points.astype(np.float32, copy=False)
    if points.shape[0] == 0:
        raise ValueError("GT point cloud is empty")
    rng = np.random.default_rng(seed)
    if points.shape[0] < count:
        ids = rng.choice(points.shape[0], size=count, replace=True)
        return points[ids]
    candidate_count = min(points.shape[0], max(count * 2, 4096))
    candidates = points[rng.choice(points.shape[0], size=candidate_count, replace=False)]
    ids = np.empty(count, dtype=np.int64)
    ids[0] = int(rng.integers(candidate_count))
    min_distance = np.full(candidate_count, np.inf, dtype=np.float32)
    for i in range(1, count):
        distance = np.square(candidates - candidates[ids[i - 1]]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
        ids[i] = int(np.argmax(min_distance))
    return candidates[ids]


class SequentialEpisodeDataset(Dataset):
    """Return contiguous windows without fabricating previous predictions from GT.

    The trainer performs model rollout through the returned frames and carries predicted
    point positions and velocities. This dataset only supplies sensor observations and GT labels.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        sequence_length: int = 4,
        frame_stride: int = 1,
        window_stride: int = 4,
        num_gt_points: int = 2048,
        image_size: int = 64,
        seed: int = 42,
        max_cached_episodes: int = 2,
        min_first_valid_ratio: float = 0.0,
        max_windows_per_episode: int | None = None,
        paths: list[Path] | None = None,
    ):
        super().__init__()
        self.paths = paths if paths is not None else split_episode_paths(data_dir, split, seed)
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.num_gt_points = num_gt_points
        self.image_size = image_size
        self.seed = seed
        self.max_cached_episodes = max_cached_episodes
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        candidate_index: list[tuple[int, int]] = []
        self._intrinsics_cache: dict[int, np.ndarray] = {}
        required_span = (sequence_length - 1) * frame_stride + 1
        for episode_id, path in enumerate(self.paths):
            intrinsic_path = path.parent / "intrinsics.npy"
            if not intrinsic_path.exists():
                raise FileNotFoundError(f"Missing camera intrinsics: {intrinsic_path}")
            with np.load(path, allow_pickle=False) as episode:
                if "gt_points" not in episode.files:
                    raise RuntimeError(f"Episode lacks gt_points: {path}")
                num_frames = int(episode["rgb"].shape[0])
                source_height, source_width = episode["rgb"].shape[1:3]
                depth = episode["depth"] if min_first_valid_ratio > 0 else None
            intrinsics = np.load(intrinsic_path).astype(np.float32).copy()
            scale_x = image_size / source_width
            scale_y = image_size / source_height
            intrinsics[0, [0, 2]] *= scale_x
            intrinsics[1, [1, 2]] *= scale_y
            self._intrinsics_cache[episode_id] = intrinsics
            for start in range(0, max(0, num_frames - required_span + 1), window_stride):
                if depth is not None:
                    first = depth[start]
                    valid_ratio = float((np.isfinite(first) & (first > 0.0) & (first <= 2.0)).mean())
                    if valid_ratio < min_first_valid_ratio:
                        continue
                candidate_index.append((episode_id, start))
        if not candidate_index:
            raise RuntimeError("No valid sequence windows were constructed")
        self._candidate_index = candidate_index
        self.max_windows_per_episode = max_windows_per_episode
        self.index: list[tuple[int, int]] = []
        self.set_epoch(0)

        self._gt_cache = {
            episode_id: _sample_gt(np.load(path, allow_pickle=False)["gt_points"], num_gt_points, seed + episode_id)
            for episode_id, path in enumerate(self.paths)
        }

    def set_epoch(self, epoch: int) -> None:
        """Deterministically resample bounded windows while covering every episode."""
        if self.max_windows_per_episode is None or self.max_windows_per_episode <= 0:
            self.index = list(self._candidate_index)
            return
        grouped: dict[int, list[int]] = {}
        for episode_id, start in self._candidate_index:
            grouped.setdefault(episode_id, []).append(start)
        rng = np.random.default_rng(self.seed + epoch * 1_000_003)
        selected: list[tuple[int, int]] = []
        for episode_id in sorted(grouped):
            starts = grouped[episode_id]
            count = min(self.max_windows_per_episode, len(starts))
            ids = rng.choice(len(starts), size=count, replace=False)
            selected.extend((episode_id, starts[int(index)]) for index in ids)
        self.index = selected

    def __len__(self) -> int:
        return len(self.index)

    def _episode(self, episode_id: int) -> dict[str, np.ndarray]:
        if episode_id in self._cache:
            value = self._cache.pop(episode_id)
            self._cache[episode_id] = value
            return value
        with np.load(self.paths[episode_id], allow_pickle=False) as archive:
            value = {key: archive[key] for key in archive.files if key != "gt_points"}
        self._cache[episode_id] = value
        while len(self._cache) > self.max_cached_episodes:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, start = self.index[index]
        episode = self._episode(episode_id)
        ids = start + np.arange(self.sequence_length) * self.frame_stride
        rgb = torch.from_numpy(episode["rgb"][ids].copy()).permute(0, 3, 1, 2).float() / 255.0
        depth = torch.from_numpy(episode["depth"][ids].copy()).permute(0, 3, 1, 2).float()
        semantic = torch.from_numpy(episode["semantic"][ids].copy()).permute(0, 3, 1, 2).long()
        pose = torch.from_numpy(episode["pose"][ids].copy()).float()
        joint = torch.from_numpy(episode["joint_pos"][ids].copy()).float()
        if "joint_vel" in episode:
            joint_velocity = torch.from_numpy(episode["joint_vel"][ids].copy()).float()
        else:
            joint_velocity = torch.zeros_like(joint)
        if "collision" in episode:
            collision = torch.from_numpy(episode["collision"][ids].copy()).bool()
        else:
            collision = torch.zeros(self.sequence_length, dtype=torch.bool)
        rgb = F.interpolate(rgb, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        depth_small = F.interpolate(depth, (self.image_size, self.image_size), mode="nearest")
        semantic_small = F.interpolate(semantic.float(), (self.image_size, self.image_size), mode="nearest").long()
        return {
            "rgb": rgb,
            "depth": depth_small,
            "depth_image": depth_small,
            "semantic": semantic_small,
            "semantic_image": semantic_small,
            "pose": pose,
            "joint_pos": joint,
            "joint_vel": joint_velocity,
            "collision": collision,
            "intrinsics": torch.from_numpy(self._intrinsics_cache[episode_id].copy()),
            "gt_points": torch.from_numpy(self._gt_cache[episode_id].copy()),
            "episode_id": torch.tensor(episode_id, dtype=torch.long),
            "start_frame": torch.tensor(start, dtype=torch.long),
        }
