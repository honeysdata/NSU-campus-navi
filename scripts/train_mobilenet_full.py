"""Train the ImageNet-pretrained MobileNetV3 U-Net on the same full dataset as SAME SAMPLE."""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compare_models import EPOCHS, THRESHOLD, MobileUNet, predict, save_overlay, scores, loss_fn
from train_unet import NSUDataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data/datasets/namseoul_university"


def main():
    torch.manual_seed(7)
    dataset = NSUDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = MobileUNet()
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
    checkpoint = DATASET_ROOT / "unet_mobilenetv3_full.pt"
    torch.save(model.state_dict(), checkpoint)
    validation = scores(model, DataLoader(dataset, batch_size=4, shuffle=False))
    mask, image = predict(model, DATASET_ROOT / "images/18_223653_102117.jpg")
    save_overlay(image, mask, DATASET_ROOT / "comparison_unet_mobilenetv3_overlay.png", "U-Net + MobileNetV3 · ImageNet", (42, 125, 219))
    result = {"epochs": EPOCHS, "threshold": THRESHOLD, "training_tiles": len(dataset), "seconds": seconds, "loss": losses, **validation, "checkpoint": checkpoint.name, "encoder": "ImageNet pretrained"}
    (DATASET_ROOT / "mobilenet_full_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
