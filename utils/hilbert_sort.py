import math
from typing import Optional, Union

import numpy as np
import torch


def _hilbert_xy2d(n: int, x: int, y: int) -> int:
    """Map grid coordinates to a Hilbert curve index on an n x n grid (n is a power of 2)."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def _normalize_coords(coords: np.ndarray, patch_size: Optional[int] = None) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f'coords must have shape [n_tokens, 2], got {coords.shape}')

    if patch_size is not None and patch_size > 0:
        coords = coords / float(patch_size)

    coords = coords - coords.min(axis=0, keepdims=True)
    return coords


def _coords_to_grid_indices(coords: np.ndarray) -> np.ndarray:
    coords = _normalize_coords(coords)
    max_val = float(max(coords[:, 0].max(), coords[:, 1].max()))
    if max_val <= 0:
        return np.zeros(len(coords), dtype=np.int64)

    order = max(1, int(math.ceil(math.log2(max_val + 1))))
    grid_size = 1 << order
    scale = (grid_size - 1) / max_val
    xs = np.clip(np.rint(coords[:, 0] * scale), 0, grid_size - 1).astype(np.int64)
    ys = np.clip(np.rint(coords[:, 1] * scale), 0, grid_size - 1).astype(np.int64)
    return np.array([_hilbert_xy2d(grid_size, int(x), int(y)) for x, y in zip(xs, ys)])


def coords_are_valid(
    coords: Union[torch.Tensor, np.ndarray, None],
    n_tokens: int,
) -> bool:
    """Return True when coords provide one (x, y) pair per token."""
    if coords is None:
        return False

    if isinstance(coords, torch.Tensor):
        if coords.numel() == 0:
            return False
        shape = tuple(coords.shape)
        if len(shape) == 3:
            return shape[1] == n_tokens and shape[2] == 2
        if len(shape) == 2:
            return shape[0] == n_tokens and shape[1] == 2
        return False

    coords = np.asarray(coords)
    if coords.size == 0:
        return False
    return coords.ndim == 2 and coords.shape[0] == n_tokens and coords.shape[1] == 2


def pseudo_grid_coords(n_tokens: int) -> np.ndarray:
    """Build row-major pseudo-grid coordinates for n_tokens before square padding."""
    side = int(math.ceil(math.sqrt(n_tokens)))
    positions = np.arange(n_tokens, dtype=np.int64)
    xs = positions % side
    ys = positions // side
    return np.stack([xs, ys], axis=1)


def hilbert_sort_indices(
    coords: Union[np.ndarray, torch.Tensor, None] = None,
    n_tokens: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Return argsort indices that order tokens along a 2D Hilbert curve.

    If coords are provided, they are normalized and mapped to Hilbert indices.
    Otherwise, a sqrt(n) pseudo-grid is used (same layout assumption as PPEG).
    """
    if coords is None:
        if n_tokens is None:
            raise ValueError('Either coords or n_tokens must be provided.')
        coords = pseudo_grid_coords(n_tokens)
    elif isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    if n_tokens is not None and len(coords) > n_tokens:
        coords = coords[:n_tokens]
    elif n_tokens is not None and len(coords) < n_tokens:
        raise ValueError(
            f'coords length ({len(coords)}) is shorter than feature length ({n_tokens}).'
        )

    if patch_size is not None:
        coords = _normalize_coords(coords, patch_size=patch_size)

    hilbert_indices = _coords_to_grid_indices(coords)
    return np.argsort(hilbert_indices, kind='stable')


def hilbert_sort_features(
    features: torch.Tensor,
    coords: Union[torch.Tensor, np.ndarray, None] = None,
    patch_size: Optional[int] = None,
) -> torch.Tensor:
    """Reorder [B, N, C] features (and optional coords) by Hilbert curve."""
    if features.ndim != 3:
        raise ValueError(f'features must have shape [B, N, C], got {tuple(features.shape)}')

    sorted_batches = []
    for batch_idx in range(features.shape[0]):
        n_tokens = features.shape[1]
        batch_coords = None
        if coords is not None:
            candidate = coords[batch_idx] if coords.ndim == 3 else coords
            if coords_are_valid(candidate, n_tokens):
                batch_coords = candidate
        order = hilbert_sort_indices(
            coords=batch_coords,
            n_tokens=n_tokens,
            patch_size=patch_size,
        )
        order = torch.as_tensor(order, device=features.device, dtype=torch.long)
        sorted_batches.append(features[batch_idx].index_select(0, order))

    return torch.stack(sorted_batches, dim=0)
