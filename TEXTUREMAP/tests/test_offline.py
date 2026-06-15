"""Offline unit tests for the engine-free core (src/).

Runs on CPU with only torch + numpy — no GPU, dataset, or checkpoint needed.

    python tests/test_offline.py      # plain runner, prints OK
    pytest tests/test_offline.py      # also works
"""
import math
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import interpolation as I
from src import seam as S
from src import projection as P
from src import viz as V


# --------------------------------------------------------------------------- #
# interpolation
# --------------------------------------------------------------------------- #
def test_slerp_endpoints():
    a = torch.randn(512)
    b = torch.randn(512)
    assert torch.allclose(I.slerp(a, b, 0.0), a, atol=1e-4)
    assert torch.allclose(I.slerp(a, b, 1.0), b, atol=1e-4)


def test_slerp_vectorized_shape():
    a, b = torch.randn(512), torch.randn(512)
    t = torch.linspace(0, 1, 50)
    out = I.slerp(a, b, t)
    assert out.shape == (50, 512)
    # endpoints of the vectorized path match the anchors
    assert torch.allclose(out[0], a, atol=1e-4)
    assert torch.allclose(out[-1], b, atol=1e-4)


def test_slerp_colinear_fallback():
    a = torch.randn(512)
    b = a * 2.0  # colinear -> sin(omega) ~ 0 -> lerp fallback, no NaN
    out = I.slerp(a, b, torch.linspace(0, 1, 10))
    assert torch.isfinite(out).all()


def test_morph_loop_is_closed():
    a, b = torch.randn(64), torch.randn(64)
    path = I.morph_loop(a, b, frames=120)
    assert path.shape == (120, 64)
    # starts near a, reaches near b at the midpoint, returns near a (loops)
    assert torch.allclose(path[0], a, atol=1e-3)
    assert torch.allclose(path[60], b, atol=1e-2)
    wrap_jump = (path[0] - path[-1]).abs().max()
    step = (path[1:] - path[:-1]).abs().max()
    assert wrap_jump <= 3 * step + 1e-4  # seam jump no worse than a normal step


def test_closed_loop_seamless_and_smooth():
    anchors = torch.randn(5, 16, 512)  # W+ layout
    path = I.closed_loop(anchors, frames=300)
    assert path.shape == (300, 16, 512)
    flat = path.reshape(300, -1)
    steps = (flat[1:] - flat[:-1]).norm(dim=1)
    wrap = (flat[0] - flat[-1]).norm()
    # closed: the wrap step is comparable to interior steps (no big cut)
    assert wrap <= 3 * steps.mean()
    # smooth: no single step is a wild outlier
    assert steps.max() <= 6 * steps.mean()


def test_closed_loop_two_anchors_delegates():
    a = torch.randn(2, 128)
    path = I.closed_loop(a, frames=80)
    assert path.shape == (80, 128)


def test_smooth_trajectory_reduces_jitter():
    base = I.closed_loop(torch.randn(4, 256), frames=200)
    noisy = base + 0.3 * torch.randn_like(base)
    sm = I.smooth_trajectory(noisy, sigma=3.0, periodic=True)
    rough = lambda p: (p[1:] - p[:-1]).abs().mean()
    assert rough(sm) < rough(noisy)
    assert sm.shape == noisy.shape


def test_kmeans_recovers_two_clusters():
    g = torch.Generator().manual_seed(1)
    c0 = torch.zeros(512) - 5
    c1 = torch.zeros(512) + 5
    x = torch.cat([c0 + torch.randn(60, 512, generator=g),
                   c1 + torch.randn(60, 512, generator=g)])
    centroids, labels = I.kmeans(x, k=2, seed=0)
    # the two recovered centroids should be far apart and near +-5
    sep = (centroids[0] - centroids[1]).norm()
    assert sep > 100  # ~ sqrt(512) * 10
    assert len(labels.unique()) == 2


def test_flatten_roundtrip():
    ws = torch.randn(7, 16, 512)
    flat, trailing = I.flatten_ws(ws)
    assert flat.shape == (7, 16 * 512)
    assert torch.equal(I.unflatten_ws(flat, trailing), ws)


# --------------------------------------------------------------------------- #
# seam
# --------------------------------------------------------------------------- #
def test_circular_pad_wraps_x():
    x = torch.arange(4 * 6, dtype=torch.float32).reshape(1, 1, 4, 6)
    p = S.circular_pad_2d(x, pad=2)
    assert p.shape == (1, 1, 8, 10)
    # left pad columns equal the wrapped right-most original columns
    core = p[:, :, 2:6, 2:8]
    assert torch.allclose(p[:, :, 2:6, 0], core[:, :, :, -2])
    assert torch.allclose(p[:, :, 2:6, 1], core[:, :, :, -1])


def test_make_tileable_matches_edges_numpy():
    rng = np.random.default_rng(0)
    img = rng.random((32, 48, 3)).astype(np.float32)
    before = np.abs(img[:, 0] - img[:, -1]).mean()
    out = S.make_horizontally_tileable(img, blend_frac=0.15)
    after = np.abs(out[:, 0] - out[:, -1]).mean()
    assert out.shape == img.shape
    assert after < before
    assert np.allclose(out[:, 0], out[:, -1], atol=1e-5)  # seam columns equal


def test_make_tileable_torch_chw():
    img = torch.rand(3, 32, 48)
    out = S.make_horizontally_tileable(img, blend_frac=0.1)
    assert out.shape == img.shape
    assert torch.allclose(out[:, :, 0], out[:, :, -1], atol=1e-5)


def test_wrap_loss_low_for_tileable():
    img = torch.rand(2, 3, 16, 24)
    tile = torch.stack([
        torch.from_numpy(
            S.make_horizontally_tileable(img[i].permute(1, 2, 0).numpy(), 0.2)
        ).permute(2, 0, 1)
        for i in range(2)
    ])
    assert S.wrap_consistency_loss(tile) < S.wrap_consistency_loss(img)


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def test_square_to_equirect_is_2to1():
    img = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
    eq = P.square_to_equirect(img, out_h=64)
    assert eq.shape == (64, 128, 3)
    assert np.isfinite(eq).all()


def test_pole_fade_runs():
    img = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
    eq = P.square_to_equirect(img, out_h=64, pole_fade=0.1)
    assert eq.shape == (64, 128, 3)
    assert np.isfinite(eq).all()


def test_sphere_preview_shape_and_finite():
    eq = np.random.default_rng(0).random((64, 128, 3)).astype(np.float32)
    sph = P.equirect_to_sphere_preview(eq, size=80, yaw=0.5, pitch=0.2)
    assert sph.shape == (80, 80, 3)
    assert np.isfinite(sph).all()


# --------------------------------------------------------------------------- #
# viz
# --------------------------------------------------------------------------- #
def test_contact_sheet_grid():
    imgs = [np.full((10, 10, 3), i / 16.0, np.float32) for i in range(16)]
    sheet = V.contact_sheet(imgs, cols=4, pad=1)
    # 4x4 grid of 10px tiles + 1px padding (5 gaps)
    assert sheet.shape == (4 * 10 + 5, 4 * 10 + 5, 3)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nALL {len(fns)} OFFLINE TESTS PASSED")


if __name__ == "__main__":
    _run_all()
