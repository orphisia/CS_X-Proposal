"""Visualization helpers — contact sheets and sphere previews for eyeball QA.

Core functions are array-in / array-out (testable with numpy alone). File I/O
wrappers (:func:`save_image`, :func:`load_image`) lazily import ``imageio`` so
importing this module never requires it.
"""
from __future__ import annotations

import numpy as np

from . import projection


def _to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0 + 1e-4:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def contact_sheet(images, cols=8, pad=2, bg=0.0):
    """Tile a list of equal-shape ``H x W x C`` arrays into one grid array.

    Returns a float array in the same value range as the inputs. Use for the
    64-image QA sheet in ``05_pick_checkpoint.py``.
    """
    imgs = [np.asarray(im, dtype=np.float32) for im in images]
    if not imgs:
        raise ValueError("no images")
    h, w = imgs[0].shape[:2]
    c = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
    imgs = [im.reshape(h, w, c) for im in imgs]

    n = len(imgs)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    sheet = np.full(
        (rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, c),
        float(bg),
        dtype=np.float32,
    )
    for i, im in enumerate(imgs):
        r, cc = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + cc * (w + pad)
        sheet[y : y + h, x : x + w] = im
    return sheet[..., 0] if c == 1 else sheet


def sphere_preview(eq, size=256, yaw=0.0, pitch=0.0, bg=0.0):
    """Thin alias for :func:`projection.equirect_to_sphere_preview`."""
    return projection.equirect_to_sphere_preview(eq, size=size, yaw=yaw, pitch=pitch, bg=bg)


# --------------------------------------------------------------------------- #
# file I/O (lazy imageio)
# --------------------------------------------------------------------------- #
def save_image(path, arr):
    import imageio.v2 as imageio

    imageio.imwrite(path, _to_uint8(arr))


def load_image(path):
    import imageio.v2 as imageio

    return np.asarray(imageio.imread(path), dtype=np.float32) / 255.0
