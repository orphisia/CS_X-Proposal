"""CSX spherical-texture interpolation — core library.

Pure torch/numpy modules. None of these import the StyleGAN engine, so they
run on CPU with only ``torch`` + ``numpy`` installed and are unit-testable
without a GPU, a dataset, or a trained checkpoint. The engine (NVlabs/stylegan3)
is only needed by the render scripts in ``scripts/``.

Modules
-------
interpolation : W-space latent walks (slerp, closed loops, smoothing, k-means).
seam          : horizontal-wrap utilities so textures tile onto a sphere.
projection    : square <-> equirectangular <-> sphere-preview mapping.
viz           : contact sheets and sphere previews for eyeball QA.
"""

__all__ = ["interpolation", "seam", "projection", "viz"]
