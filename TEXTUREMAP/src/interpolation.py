"""Latent-space interpolation for StyleGAN W-walks.

Interpolate in **W space** (the intermediate latent after the mapping network),
not Z — W gives StyleGAN its famously clean morphs. Use **slerp** for pairwise
moves so motion speed stays even on the manifold, and a periodic spline for
multi-anchor loops so the rendered video can play forever with no cut.

This module is engine-free (only torch + numpy) so it is unit-testable on CPU.

Shape conventions
-----------------
Anchors are W-space vectors. We interpolate them as flat ``[K, D]`` tensors
(``D == 512`` for a single broadcast style, or ``D == num_ws * 512`` for W+).
Use :func:`flatten_ws` / :func:`unflatten_ws` to move between StyleGAN's
``[N, num_ws, 512]`` layout and the flat ``[N, D]`` used here.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# shape helpers
# --------------------------------------------------------------------------- #
def flatten_ws(ws):
    """``[N, num_ws, C] -> ([N, num_ws*C], trailing_shape)``.

    Already-flat ``[N, D]`` input is returned unchanged with ``trailing=None``.
    """
    ws = torch.as_tensor(ws)
    if ws.ndim == 2:
        return ws, None
    n = ws.shape[0]
    trailing = tuple(ws.shape[1:])
    return ws.reshape(n, -1), trailing


def unflatten_ws(flat, trailing):
    """Inverse of :func:`flatten_ws`."""
    if trailing is None:
        return flat
    return flat.reshape(flat.shape[0], *trailing)


# --------------------------------------------------------------------------- #
# pairwise interpolation
# --------------------------------------------------------------------------- #
def lerp(a, b, t):
    """Linear interpolation. ``t`` scalar -> ``[D]``; ``t`` 1-D -> ``[T, D]``."""
    a = torch.as_tensor(a).float()
    b = torch.as_tensor(b).float()
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    scalar = t.ndim == 0
    tt = t.reshape(-1, 1)
    out = (1 - tt) * a + tt * b
    return out[0] if scalar else out


def slerp(a, b, t, eps=1e-6):
    """Spherical linear interpolation between vectors ``a``, ``b`` (shape ``[D]``).

    ``t`` may be a scalar (returns ``[D]``) or a 1-D tensor/array (returns
    ``[T, D]``). Falls back to :func:`lerp` when ``a`` and ``b`` are nearly
    colinear (``sin(omega) ~ 0``), which keeps the function well-defined at the
    degenerate endpoints.
    """
    a = torch.as_tensor(a).float()
    b = torch.as_tensor(b).float()
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    scalar = t.ndim == 0
    tt = t.reshape(-1, 1)  # [T, 1]

    a_n = a / (a.norm() + eps)
    b_n = b / (b.norm() + eps)
    dot = (a_n * b_n).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    if so.abs() < eps:
        out = (1 - tt) * a + tt * b
    else:
        out = (torch.sin((1 - tt) * omega) / so) * a + (torch.sin(tt * omega) / so) * b
    return out[0] if scalar else out


# --------------------------------------------------------------------------- #
# looping paths
# --------------------------------------------------------------------------- #
def morph_loop(a, b, frames, mode="slerp"):
    """The hero two-anchor motion: ``a -> b -> a``, smooth and seamless.

    Uses a cosine schedule ``tau = (1 - cos(2*pi*s)) / 2`` (``s`` in ``[0, 1)``)
    so the walk eases in/out at both ends, pauses momentarily on each anchor,
    and returns exactly to the start — perfect for an installation loop.

    Returns ``[frames, D]``.
    """
    a = torch.as_tensor(a).float()
    b = torch.as_tensor(b).float()
    s = torch.arange(frames, dtype=torch.float32) / max(frames, 1)
    tau = (1 - torch.cos(2 * math.pi * s)) / 2  # 0 -> 1 -> 0, C1 periodic
    interp = slerp if mode == "slerp" else lerp
    return torch.stack([interp(a, b, float(ti)) for ti in tau])


def _catmull_rom_segment(p0, p1, p2, p3, u):
    """Centripetal-style Catmull-Rom on one segment ``p1 -> p2``.

    ``u`` in ``[0, 1)`` shape ``[m]``; points shape ``[D]``; returns ``[m, D]``.
    """
    u = u.reshape(-1, 1)
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
    )


def closed_loop(anchors, frames):
    """Smooth (C1) closed periodic spline through ``anchors`` -> ``[frames, ...]``.

    A periodic Catmull-Rom curve: each segment's tangents use the neighbouring
    anchors with wraparound, so the path is C1-continuous *and* returns exactly
    to its start (the video loops with no velocity pulse at the seam). Trailing
    anchor dimensions (e.g. ``[K, num_ws, 512]``) are preserved.

    With 2 anchors this delegates to :func:`morph_loop` (a there-and-back morph).
    """
    A, trailing = flatten_ws(torch.as_tensor(anchors).float())
    K = A.shape[0]
    if K < 2:
        raise ValueError("closed_loop needs >= 2 anchors")
    if K == 2:
        path = morph_loop(A[0], A[1], frames, mode="lerp")
        return unflatten_ws(path, trailing)

    # distribute frames across the K segments (remainder spread to the front)
    per = [frames // K] * K
    for i in range(frames - sum(per)):
        per[i] += 1

    segments = []
    for i in range(K):
        p0, p1, p2, p3 = A[(i - 1) % K], A[i % K], A[(i + 1) % K], A[(i + 2) % K]
        m = per[i]
        if m == 0:
            continue
        u = torch.arange(m, dtype=A.dtype) / m  # exclude 1.0 so anchors aren't duplicated
        segments.append(_catmull_rom_segment(p0, p1, p2, p3, u))
    path = torch.cat(segments, dim=0)  # [frames, D]
    return unflatten_ws(path, trailing)


def smooth_trajectory(path, sigma=2.0, periodic=True):
    """Gaussian low-pass along the time axis (axis 0) to kill frame flicker.

    ``periodic=True`` wraps the kernel (use for closed loops so the seam stays
    smooth). ``sigma <= 0`` is a no-op. Trailing dims are preserved.
    """
    if sigma <= 0:
        return torch.as_tensor(path).float()
    p = torch.as_tensor(path).float()
    T = p.shape[0]
    radius = max(1, int(round(3 * sigma)))
    k = torch.arange(-radius, radius + 1, dtype=torch.float32)
    w = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    w = w / w.sum()

    flat = p.reshape(T, -1)  # [T, D]
    offsets = k.long()
    if periodic:
        idx = (torch.arange(T)[:, None] + offsets[None, :]) % T
    else:
        idx = (torch.arange(T)[:, None] + offsets[None, :]).clamp(0, T - 1)
    gathered = flat[idx]  # [T, 2r+1, D]
    out = (gathered * w[None, :, None]).sum(dim=1)  # [T, D]
    return out.reshape_as(p)


# --------------------------------------------------------------------------- #
# anchor discovery
# --------------------------------------------------------------------------- #
def kmeans(x, k, iters=50, seed=0):
    """Plain Lloyd's k-means. ``x: [N, D] -> (centroids [k, D], labels [N])``.

    Used to recover the mandala vs. icon centroids post-hoc from a batch of W
    vectors (the unconditional-then-cluster trick from the build plan §3), giving
    you two interpolation endpoints with no retraining.
    """
    x = torch.as_tensor(x).float()
    n = x.shape[0]
    if k > n:
        raise ValueError(f"k={k} > N={n}")
    g = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(n, generator=g)[:k]].clone()
    labels = torch.full((n,), -1, dtype=torch.long)
    for it in range(iters):
        new_labels = torch.cdist(x, centroids).argmin(dim=1)
        if it > 0 and torch.equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = labels == j
            if members.any():
                centroids[j] = x[members].mean(dim=0)
    return centroids, labels
