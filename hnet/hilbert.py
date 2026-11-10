"""
2D Hilbert Space-Filling Curve utilities for mapping 2D grids (H, W) <-> 1D sequences (L = H * W)
while preserving 2D spatial locality:
  - Every step t -> t+1 along the 1D Hilbert sequence is an exact 4-connected neighbor on the 2D grid.
  - Every contiguous 1D chunk [t_{k-1}+1 .. t_k] corresponds to a compact 2D spatial region.
"""

from typing import Tuple
import torch


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert 1D Hilbert distance d in [0, n*n - 1] to 2D (x, y) in [0, n - 1]^2."""
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def build_hilbert_lut(grid_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build lookup tables for a square power-of-2 2D grid of size (grid_size, grid_size).

    Returns:
        hilbert_to_raster: (L,) long tensor -> for each Hilbert index t, gives raster index y * W + x
        raster_to_hilbert: (L,) long tensor -> inverse permutation (for each raster index, gives Hilbert t)
        pos_xy_norm: (L, 2) float32 tensor -> normalized (x, y) coordinates in [-1, 1]^2 in Hilbert order
        pos_xy_grid: (L, 2) float32 tensor -> grid (x, y) coordinates in [0, grid_size - 1]^2 in Hilbert order
    """
    assert grid_size > 0 and (grid_size & (grid_size - 1)) == 0, (
        f"grid_size must be a power of 2 for 2D Hilbert curve, got {grid_size}"
    )
    L = grid_size * grid_size
    hilbert_to_raster = torch.empty(L, dtype=torch.long)
    raster_to_hilbert = torch.empty(L, dtype=torch.long)
    pos_xy_grid = torch.empty((L, 2), dtype=torch.float32)

    for d in range(L):
        x, y = d2xy(grid_size, d)
        r_idx = y * grid_size + x
        hilbert_to_raster[d] = r_idx
        raster_to_hilbert[r_idx] = d
        pos_xy_grid[d, 0] = float(x)
        pos_xy_grid[d, 1] = float(y)

    # Normalized coordinates in [-1, 1] (cell centers)
    pos_xy_norm = (2.0 * (pos_xy_grid + 0.5) / float(grid_size)) - 1.0
    return hilbert_to_raster, raster_to_hilbert, pos_xy_norm, pos_xy_grid
