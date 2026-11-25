"""Trains the TinyUNet3D bone/background segmenter on on-the-fly synthetic
CT volumes, then evaluates Dice on held-out synthetic samples.

Run:
    python3 src/train_segmentation.py
Outputs:
    outputs/unet3d_bone.pt       -- trained weights
    outputs/training_curve.png   -- loss/dice curves
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_ct import generate_sample
from unet3d import TinyUNet3D, dice_loss, dice_score

VOL_SIZE = 64
DEVICE = "cpu"


def make_batch(n, seed0, with_defect=True):
    vols, masks = [], []
    for i in range(n):
        s = generate_sample(size=VOL_SIZE, seed=seed0 + i, with_defect=with_defect)
        v = s.volume
        v = (v - v.mean()) / (v.std() + 1e-6)
        vols.append(v[None])
        masks.append(s.bone_mask.astype(np.float32)[None])
    x = torch.tensor(np.stack(vols), dtype=torch.float32)
    y = torch.tensor(np.stack(masks), dtype=torch.float32)
    return x, y


def main(epochs=12, steps_per_epoch=6, batch_size=2, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = TinyUNet3D(in_ch=1, base=8).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    history = {"loss": [], "dice": []}
    t0 = time.time()
    rng_seed = 1000
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for step in range(steps_per_epoch):
            x, y = make_batch(batch_size, seed0=rng_seed)
            rng_seed += batch_size
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        # quick held-out eval
        model.eval()
        with torch.no_grad():
            xv, yv = make_batch(2, seed0=50000 + epoch)
            logits = model(xv)
            pred = (torch.sigmoid(logits) > 0.5).float()
            d = dice_score(pred, yv)

        history["loss"].append(float(np.mean(epoch_losses)))
        history["dice"].append(d)
        print(f"epoch {epoch+1:02d}/{epochs}  loss={history['loss'][-1]:.4f}  "
              f"val_dice={d:.4f}  elapsed={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), "outputs/unet3d_bone.pt")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(history["loss"], marker="o")
    axes[0].set_title("Training loss (BCE + Dice)")
    axes[0].set_xlabel("epoch")
    axes[1].plot(history["dice"], marker="o", color="darkgreen")
    axes[1].set_title("Held-out Dice score")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("outputs/training_curve.png", dpi=140)
    print("Saved outputs/unet3d_bone.pt and outputs/training_curve.png")
    print(f"Final held-out Dice: {history['dice'][-1]:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
