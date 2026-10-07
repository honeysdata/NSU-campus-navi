"""Minimal U-Net training entry point for the real NSU tile dataset.

Install: pip install torch torchvision pillow
Run:     python3 train_unet.py
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data/datasets/namseoul_university"


class NSUDataset(Dataset):
    def __init__(self):
        self.masks = sorted((ROOT / "masks").glob("*.pgm"))
        self.images = [ROOT / "images" / (p.stem + ".jpg") for p in self.masks]

    def __len__(self): return len(self.masks)

    def __getitem__(self, i):
        image = np.asarray(Image.open(self.images[i]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.masks[i]).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None])


class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.e1 = DoubleConv(3, 32); self.e2 = DoubleConv(32, 64); self.pool = nn.MaxPool2d(2); self.up = nn.ConvTranspose2d(64, 32, 2, stride=2); self.out = nn.Conv2d(32, 1, 1); self.d = DoubleConv(64, 32)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.pool(a)); c = self.up(b); return self.out(self.d(torch.cat([c, a], 1)))


def main():
    ds = NSUDataset()
    if not ds: raise SystemExit("데이터셋이 비어 있습니다. 먼저 build_nsu_dataset.py를 실행하세요.")
    loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True)
    started = time.perf_counter()
    model = UNet(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Walkable pixels are sparse; plain BCE can learn the all-background solution.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))
    for epoch in range(1, 11):
        total = 0.0
        for x, y in loader:
            logits = model(x)
            bce = loss_fn(logits, y)
            p = torch.sigmoid(logits)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = bce + dice
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        print(f"epoch={epoch:02d} loss={total / len(loader):.4f}")
    elapsed = time.perf_counter() - started
    torch.save(model.state_dict(), ROOT / "unet_nsu_walkable.pt")
    (ROOT / "training_metrics.json").write_text(json.dumps({"epochs": 10, "seconds": round(elapsed, 1), "seconds_per_epoch": round(elapsed / 10, 1)}, indent=2), encoding="utf-8")
    print(f"training_seconds={elapsed:.1f} seconds_per_epoch={elapsed / 10:.1f}")


if __name__ == "__main__": main()
