"""Texture <-> sphere (equirectangular) mapping helpers.

Equirectangular convention (the 2:1 "unfolded world map" sphere UV):

* width ``W = 2 * H``
* ``x in [0, W)``  -> longitude ``[0, 360)``
* ``y in [0, H)``  -> latitude ``[+90 (top), -90 (bottom)]``

Engine-free numpy; resizing uses torch so no OpenCV/PIL dependency in the core.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _resize(img, out_h, out_w):
    """Bilinear resize of ``H x W x C`` (or ``H x W``) numpy -> ``out_h x out_w``."""
    arr = np.asarray(img).astype(np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[..., None]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    out = t.squeeze(0).permute(1, 2, 0).numpy()
    return out[..., 0] if squeeze else out


def _apply_pole_fade(eq, pole_fade):
    """Soften detail toward the poles by blending rows into their row-mean.

    ``pole_fade`` in ``(0, 1]`` is the fraction of the image height affected at
    *each* pole. Pole pinching on a sphere reads as acceptable swirl for an
    abstract moving texture (build plan §4); this just keeps it from looking torn.
    """
    eq = eq.copy()
    H = eq.shape[0]
    band = max(1, int(round(H * pole_fade)))
    row_mean = eq.reshape(H, -1).mean(axis=1) if eq.ndim == 2 else eq.mean(axis=1, keepdims=True)
    for i in range(band):
        alpha = (i / band)  # 0 at the very pole -> 1 at the band edge
        for y in (i, H - 1 - i):
            eq[y] = alpha * eq[y] + (1 - alpha) * row_mean[y]
    return eq


def square_to_equirect(img, out_h=512, pole_fade=0.0):
    """Map a (horizontally-tileable) square texture to a 2:1 equirect image.

    The source spans 360 deg in x, so this is a resize to ``out_h x 2*out_h``
    with optional pole softening. For a seamless wrap the source should already
    be tileable (see :func:`seam.make_horizontally_tileable`).
    """
    eq = _resize(img, out_h, 2 * out_h)
    if pole_fade > 0:
        eq = _apply_pole_fade(eq, pole_fade)
    return eq


def equirect_to_sphere_preview(eq, size=256, yaw=0.0, pitch=0.0, bg=0.0):
    """Render an orthographic preview of ``eq`` mapped onto a sphere (for QA).

    ``yaw`` / ``pitch`` rotate the globe (radians). Pixels outside the disc get
    ``bg``. Nearest-neighbour sampling — this is a sanity check, not the final
    render. Returns ``size x size x C`` float.
    """
    eq = np.asarray(eq).astype(np.float32)
    if eq.ndim == 2:
        eq = eq[..., None]
    H, W, C = eq.shape

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = (xs - size / 2) / (size / 2)
    ny = (ys - size / 2) / (size / 2)
    r2 = nx ** 2 + ny ** 2
    mask = r2 <= 1.0
    nz = np.sqrt(np.clip(1 - r2, 0, 1))

    # camera looks down -z; visible front hemisphere
    X, Y, Z = nx, -ny, nz
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # yaw about Y
    Xr = cy * X + sy * Z
    Zr = -sy * X + cy * Z
    Yr = Y
    # pitch about X
    Yr2 = cp * Yr - sp * Zr
    Zr2 = sp * Yr + cp * Zr
    Xr2 = Xr

    lon = np.arctan2(Xr2, Zr2)               # [-pi, pi]
    lat = np.arcsin(np.clip(Yr2, -1, 1))     # [-pi/2, pi/2]
    u = (lon / (2 * np.pi) + 0.5) % 1.0
    v = np.clip(0.5 - lat / np.pi, 0, 1)     # 0 top -> 1 bottom

    sx = np.clip((u * W).astype(np.int64), 0, W - 1)
    sy_ = np.clip((v * H).astype(np.int64), 0, H - 1)
    sampled = eq[sy_, sx]                    # [size, size, C]

    out = np.full((size, size, C), float(bg), dtype=np.float32)
    out[mask] = sampled[mask]
    return out
