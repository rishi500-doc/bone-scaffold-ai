"""Fine-tunes the synthetic-pretrained TinyUNet3D on the one real CT sample
(TotalSegmentator s1405, left femur) to close the sim-to-real gap.

Only one real volume is available, so this is intentionally light-touch:
a handful of augmented views (small translations, flips, intensity jitter)
of that single real case, mixed with fresh synthetic batches so the model
doesn't forget the general bone/background boundary it already learned.
This demonstrates domain-adaptation reasoning, not a claim of a validated
real-data model -- the report says exactly that.

Run:
    python3 src/finetune_real.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from synth_ct import generate_sample
from real_ct import load_real_sample
from unet3d import TinyUNet3D, dice_loss, dice_score

VOL_SIZE = 64


def augment_real(sample, rng):
    vol = sample.volume.copy()
    mask = (sample.bone_mask | sample.defect_mask).astype(np.float32)
    shift = rng.integers(-4, 5, size=3)
    vol = np.roll(vol, shift, axis=(0, 1, 2))
    mask = np.roll(mask, shift, axis=(0, 1, 2))
    if rng.random() < 0.5:
        vol = np.flip(vol, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    vol = vol + rng.normal(0, 15, size=vol.shape).astype(np.float32)
    v = (vol - vol.mean()) / (vol.std() + 1e-6)
    return v[None], mask[None]


def synth_batch(n, seed0, rng=None):
    vols, masks = [], []
    for i in range(n):
        s = generate_sample(size=VOL_SIZE, seed=seed0 + i, with_defect=True)
        v = (s.volume - s.volume.mean()) / (s.volume.std() + 1e-6)
        vols.append(v[None])
        masks.append(s.bone_mask.astype(np.float32)[None])
    return vols, masks


def main(steps=40, lr=5e-4, real_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    model = TinyUNet3D(in_ch=1, base=8)
    model.load_state_dict(torch.load("outputs/unet3d_bone.pt", map_location="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    real_sample = load_real_sample(out_size=VOL_SIZE, seed=0)
    real_gt = (real_sample.bone_mask | real_sample.defect_mask)

    seed0 = 777000
    for step in range(steps):
        model.train()
        vols, masks = [], []
        n_real = 1 if rng.random() < real_frac else 0
        for _ in range(n_real):
            v, m = augment_real(real_sample, rng)
            vols.append(v); masks.append(m)
        n_synth = 2 - n_real
        sv, sm = synth_batch(n_synth, seed0); seed0 += n_synth
        vols += sv; masks += sm

        x = torch.tensor(np.stack(vols), dtype=torch.float32)
        y = torch.tensor(np.stack(masks), dtype=torch.float32)
        opt.zero_grad()
        logits = model(x)
        loss = bce(logits, y) + dice_loss(logits, y)
        loss.backward()
        opt.step()

        if (step + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                v = (real_sample.volume - real_sample.volume.mean()) / (real_sample.volume.std() + 1e-6)
                logits = model(torch.tensor(v[None, None], dtype=torch.float32))
                pred = (torch.sigmoid(logits) > 0.5).squeeze().numpy().astype(bool)
                d = 2 * (pred & real_gt).sum() / (pred.sum() + real_gt.sum())
            print(f"step {step+1:3d}/{steps}  loss={loss.item():.4f}  real_dice={d:.4f}")

    torch.save(model.state_dict(), "outputs/unet3d_bone_finetuned.pt")
    print("Saved outputs/unet3d_bone_finetuned.pt")


if __name__ == "__main__":
    main()
