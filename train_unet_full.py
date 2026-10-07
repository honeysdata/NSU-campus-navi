"""Rebuild the SAME SAMPLE U-Net checkpoint with the benchmark protocol."""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compare_models import EPOCHS, loss_fn
from train_unet import NSUDataset, UNet

ROOT = Path(__file__).parent
DATASET_ROOT = ROOT / "nsu_real_dataset"


def main():
    torch.manual_seed(7)
    dataset = NSUDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    started = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in loader:
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(round(total / len(loader), 4))
    seconds = round(time.perf_counter() - started, 1)
    torch.save(model.state_dict(), DATASET_ROOT / "unet_nsu_walkable.pt")
    (DATASET_ROOT / "training_metrics.json").write_text(json.dumps({"epochs": EPOCHS, "seconds": seconds, "seconds_per_epoch": round(seconds / EPOCHS, 1), "training_tiles": len(dataset)}, indent=2), encoding="utf-8")
    print({"seconds": seconds, "loss": losses, "checkpoint": "unet_nsu_walkable.pt", "training_tiles": len(dataset)})


if __name__ == "__main__":
    main()
