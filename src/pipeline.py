"""End-to-end pipeline: synthetic CT -> trained CNN bone segmentation ->
gyroid scaffold design -> biomechanical sweep -> STL export + figures.

Run:
    python3 src/pipeline.py --seed 7
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_ct import generate_sample
from unet3d import TinyUNet3D
from scaffold_gyroid import build_scaffold, mask_to_mesh, export_stl
from biomechanics import sweep


def run_model_segmentation(volume: np.ndarray, weights_path="outputs/unet3d_bone.pt") -> np.ndarray:
    model = TinyUNet3D(in_ch=1, base=8)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    v = (volume - volume.mean()) / (volume.std() + 1e-6)
    x = torch.tensor(v[None, None], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).squeeze().numpy().astype(bool)
    return pred


def main(seed=7, size=64, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    sample = generate_sample(size=size, seed=seed, with_defect=True)

    # "ground truth" bone region = bone_mask + defect_mask (what the CT bone
    # actually occupies before the defect); the CNN is only trained to find
    # bone vs not-bone, so we recover the full original bone silhouette as
    # the region the scaffold needs to key into, and treat the CNN's
    # prediction as the clinically-observed (post-injury) bone mask.
    pred_bone_mask = run_model_segmentation(sample.volume)
    gt_bone_mask = sample.bone_mask

    dice = 2 * (pred_bone_mask & gt_bone_mask).sum() / (pred_bone_mask.sum() + gt_bone_mask.sum())
    print(f"Inference Dice vs ground truth: {dice:.4f}")

    rows, pick = sweep(pred_bone_mask, sample.defect_mask, sample.voxel_mm)
    print("Picked scaffold: period_vox=%.1f  porosity=%.3f  E/E_solid=%.4f" %
          (pick.period_vox, pick.porosity, pick.rel_stiffness_gibson_ashby))

    sc = build_scaffold(pred_bone_mask, sample.defect_mask,
                         period_vox=pick.period_vox, gyroid_thresh=pick.gyroid_thresh)
    mesh = mask_to_mesh(sc.solid_mask, voxel_mm=sample.voxel_mm)
    export_stl(mesh, f"{outdir}/scaffold_pipeline.stl")

    # --- figures ---
    mid = size // 2
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].imshow(sample.volume[mid], cmap="gray")
    axes[0].set_title("Synthetic CT slice")
    axes[1].imshow(pred_bone_mask[mid], cmap="bone")
    axes[1].set_title(f"CNN-predicted bone mask\n(Dice={dice:.3f})")
    axes[2].imshow(sc.solid_mask[mid], cmap="viridis")
    axes[2].set_title("Scaffold (shell + gyroid infill)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{outdir}/pipeline_slices.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(5, 3.6))
    periods = [r.period_vox for r in rows]
    porosities = [r.porosity for r in rows]
    stiff = [r.rel_stiffness_gibson_ashby for r in rows]
    ax.plot(periods, porosities, "o-", label="Porosity")
    ax2 = ax.twinx()
    ax2.plot(periods, stiff, "s--", color="firebrick", label="E/E_solid (Gibson-Ashby)")
    ax.set_xlabel("Gyroid period (voxels)")
    ax.set_ylabel("Porosity")
    ax2.set_ylabel("Relative stiffness")
    ax.scatter([pick.period_vox], [pick.porosity], color="black", zorder=5, label="picked")
    fig2.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig2.tight_layout()
    fig2.savefig(f"{outdir}/porosity_stiffness_tradeoff.png", dpi=150, bbox_inches="tight")

    print(f"Saved: {outdir}/scaffold_pipeline.stl, pipeline_slices.png, porosity_stiffness_tradeoff.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    main(seed=args.seed, size=args.size)
