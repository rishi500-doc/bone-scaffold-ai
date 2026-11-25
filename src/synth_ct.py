"""
Synthetic CT volume generator for a non-paired long-bone segment with a
focal defect (e.g. a traumatic / resection defect needing reconstruction).

Why synthetic data: real patient CT of non-paired bones (clavicle, pelvis,
etc.) with defects is not freely redistributable. Procedurally generating
paired (CT-intensity-volume, ground-truth bone mask) samples is a standard
way to bootstrap and validate a segmentation pipeline before fine-tuning on
a real, access-controlled clinical dataset (e.g. the RPS project's own
patient CTs) -- the geometry (cortical shell + trabecular core + tapering
shaft + a punched-out defect) mimics the intensity/shape statistics of a
real bone CT closely enough to train and sanity-check the pipeline below.

Each sample is a cube volume of side `size` voxels with:
  - a curved, tapering shaft (simulates a long bone / rib / clavicle)
  - a dense cortical shell (high HU) and a spongier trabecular core (mid HU,
    textured with band-limited noise)
  - soft tissue background (low HU) with Gaussian scanner noise
  - a randomly placed spherical/ellipsoidal "defect" void cut out of the
    shaft -- this is the region the scaffold is designed to fill
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from dataclasses import dataclass


@dataclass
class SynthSample:
    volume: np.ndarray        # float32, HU-like intensities, shape (D,H,W)
    bone_mask: np.ndarray     # bool ground-truth bone mask
    defect_mask: np.ndarray   # bool mask of the punched-out defect region
    voxel_mm: float           # isotropic voxel size in mm (for metric scaling)


def _fractal_noise(shape, rng, octaves=4, persistence=0.55):
    noise = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    total_amp = 0.0
    for o in range(octaves):
        scale = 2 ** o
        small_shape = tuple(max(1, s // (scale * 2)) for s in shape)
        small = rng.standard_normal(small_shape).astype(np.float32)
        zoom_factors = tuple(s / ss for s, ss in zip(shape, small_shape))
        layer = ndi.zoom(small, zoom_factors, order=3)
        # zoom can over/undershoot by a voxel; crop/pad to exact shape
        layer = layer[tuple(slice(0, s) for s in shape)]
        if layer.shape != shape:
            pad = [(0, s - ls) for s, ls in zip(shape, layer.shape)]
            layer = np.pad(layer, pad, mode="edge")
        noise += amp * layer
        total_amp += amp
        amp *= persistence
    return noise / total_amp


def generate_sample(size: int = 96, seed: int | None = None,
                     with_defect: bool = True, voxel_mm: float = 0.4) -> SynthSample:
    rng = np.random.default_rng(seed)
    d = h = w = size
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w),
        indexing="ij",
    )

    # Gentle S-curve for the shaft centerline (mimics a non-straight long bone)
    curve_amp = rng.uniform(0.08, 0.18)
    cx = curve_amp * np.sin(zz * np.pi)
    cy = curve_amp * 0.6 * np.cos(zz * np.pi * 0.7)

    # Shaft radius tapers toward both ends (diaphysis thinner than epiphyses)
    base_r = rng.uniform(0.34, 0.42)
    taper = 1.0 - 0.35 * np.cos(zz * np.pi)  # wider at z=+-1, narrower mid-shaft
    radius = base_r * taper

    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    shaft = r <= radius

    # Hollow medullary canal inside the shaft
    canal_r = radius * rng.uniform(0.42, 0.55)
    canal = r <= canal_r
    cortical_shell = shaft & ~canal

    bone_mask = shaft.copy()

    # Intensity field (HU-like, arbitrary units ~ [-200, 1800])
    vol = rng.normal(loc=20, scale=15, size=(d, h, w)).astype(np.float32)  # soft tissue/background
    vol[cortical_shell] = rng.normal(1300, 80, size=cortical_shell.sum())
    trabecular = canal & shaft
    tex = _fractal_noise((d, h, w), rng, octaves=4)
    vol[trabecular] = 350 + 220 * tex[trabecular]

    defect_mask = np.zeros_like(bone_mask)
    if with_defect:
        # place a spherical defect centered somewhere along the shaft
        z0 = rng.uniform(-0.45, 0.45)
        defect_r = rng.uniform(0.12, 0.2)
        # offset slightly off-centerline so the defect bites into the cortex unevenly
        off_x = rng.uniform(-0.05, 0.05)
        off_y = rng.uniform(-0.05, 0.05)
        dist = np.sqrt((xx - (cx[d // 2, 0, 0] if False else 0) - off_x) ** 2 +
                       (yy - off_y) ** 2 + (zz - z0) ** 2)
        defect = (dist <= defect_r) & bone_mask
        defect_mask = defect
        bone_mask = bone_mask & ~defect
        vol[defect] = rng.normal(20, 10, size=defect.sum())

    vol += rng.normal(0, 25, size=vol.shape).astype(np.float32)  # scanner noise
    vol = ndi.gaussian_filter(vol, sigma=0.6)

    return SynthSample(volume=vol.astype(np.float32),
                        bone_mask=bone_mask.astype(bool),
                        defect_mask=defect_mask.astype(bool),
                        voxel_mm=voxel_mm)


if __name__ == "__main__":
    s = generate_sample(size=96, seed=0)
    print("volume", s.volume.shape, s.volume.min(), s.volume.max())
    print("bone voxels", s.bone_mask.sum(), "defect voxels", s.defect_mask.sum())
