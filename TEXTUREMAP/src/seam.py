"""Horizontal-wrap (seam) utilities for sphere-ready textures.

A flat texture wrapped on a sphere shows a vertical seam where longitude 0 meets
360. The build plan's strategy ladder:

* **(A)** train normally, hide the seam post-hoc  -> :func:`make_horizontally_tileable`
* **(B)** seam-aware training (circular conv padding) -> :func:`circular_pad_2d`,
  :func:`patch_synthesis_for_wrap`
* **(C)** native equirect training with a wrap loss -> :func:`wrap_consistency_loss`

Image tensors are ``[N, C, H, W]`` (torch) unless noted.
:func:`make_horizontally_tileable` also accepts ``H x W x C`` numpy arrays.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# (B) seam-aware convolution padding
# --------------------------------------------------------------------------- #
def circular_pad_2d(x, pad, vertical="reflect"):
    """Pad ``[N, C, H, W]`` circularly along x (wrap) and ``vertical`` along y.

    This is the primitive for strategy (B): the x-axis wraps so the generator
    learns left/right continuity natively, while y (latitude) does not wrap.
    ``vertical`` is any ``F.pad`` mode: ``reflect``, ``replicate``, ``circular``,
    or ``constant``.
    """
    x = torch.as_tensor(x)
    added_batch = x.ndim == 3
    if added_batch:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("expected [N, C, H, W] or [C, H, W]")
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")          # left/right wrap
    x = F.pad(x, (0, 0, pad, pad), mode=vertical)            # top/bottom
    return x.squeeze(0) if added_batch else x


def patch_synthesis_for_wrap(module):
    """Best-effort: switch a generator's ``nn.Conv2d`` layers to circular padding.

    Returns the number of layers patched. Use to retrofit horizontal tiling onto
    a model assembled from stock ``nn.Conv2d``.

    LIMITATION — read before relying on this. ``padding_mode='circular'`` wraps
    *both* axes, but a sphere texture must wrap only x (longitude), not y
    (latitude). And NVlabs/stylegan3 does not use ``nn.Conv2d`` for synthesis —
    it uses a custom ``conv2d_resample`` op, which this will not touch. For a
    correct strategy-(B) build, edit the stylegan2 synthesis conv to call
    :func:`circular_pad_2d` (wrap x, reflect y) before the conv and set its
    internal padding to 0. This helper is a convenience for stock-Conv2d models
    and a smoke-test stand-in only.
    """
    import torch.nn as nn

    patched = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            pad = m.padding if isinstance(m.padding, tuple) else (m.padding,)
            if any(p > 0 for p in pad) and m.padding_mode != "circular":
                m.padding_mode = "circular"
                patched += 1
    return patched


# --------------------------------------------------------------------------- #
# (A) post-hoc tileable blend
# --------------------------------------------------------------------------- #
def make_horizontally_tileable(img, blend_frac=0.1):
    """Cross-fade the left/right seam so a finished frame tiles horizontally.

    Accepts ``H x W x C`` numpy or ``[C, H, W]`` torch (returns the same type
    and shape). Columns equidistant from the wrap boundary are pulled toward
    each other, strongest at the seam (weight 0.5 -> the boundary columns become
    equal, giving C0 continuity) and fading to the original by ``blend_frac * W``
    columns in. A safety net for strategies (A)/(B).
    """
    is_torch = isinstance(img, torch.Tensor)
    if is_torch:
        chw = img.dim() == 3 and img.shape[0] in (1, 3, 4)
        arr = img.permute(1, 2, 0).cpu().numpy() if chw else img.cpu().numpy()
    else:
        arr = np.asarray(img)
        chw = False
    arr = arr.astype(np.float32, copy=True)

    W = arr.shape[1]
    b = max(1, int(round(W * blend_frac)))
    src = arr.copy()
    for j in range(b):
        w = 0.5 * (1 - j / b)  # 0.5 at the seam (j=0) -> 0 at j=b
        lc, rc = j, W - 1 - j
        left, right = src[:, lc], src[:, rc]
        arr[:, lc] = (1 - w) * left + w * right
        arr[:, rc] = (1 - w) * right + w * left

    if is_torch:
        out = torch.from_numpy(arr)
        return out.permute(2, 0, 1) if chw else out
    return arr


# --------------------------------------------------------------------------- #
# (C) training-time wrap loss
# --------------------------------------------------------------------------- #
def wrap_consistency_loss(img, width=1):
    """L1 mismatch between the left and right edges of ``[N, C, H, W]``.

    Add to the generator loss for strategy (C) so the model learns to make the
    longitude-0 edge continue seamlessly from the longitude-360 edge. Zero for a
    perfectly tileable image.
    """
    img = torch.as_tensor(img).float()
    left = img[..., :width]
    right = img[..., -width:]
    return (left - right).abs().mean()
