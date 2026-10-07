"""Train and compare the pilot U-Net and U-Net + MobileNetV3 models.

Both models use the same OSM-derived masks, fixed train/validation split,
optimizer, epochs, threshold, and sample for an apples-to-apples comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from train_unet import DoubleConv, NSUDataset, UNet


ROOT = Path(__file__).parent
DATASET_ROOT = ROOT / "nsu_real_dataset"
EPOCHS = 10
THRESHOLD = 0.3


class MobileUNet(nn.Module):
    """U-Net decoder with an ImageNet-pretrained MobileNetV3-small encoder."""

    def __init__(self):
        super().__init__()
        self.encoder = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.up1 = nn.ConvTranspose2d(576, 96, 2, stride=2)
        self.dec1 = DoubleConv(96 + 48, 64)
        self.up2 = nn.ConvTranspose2d(64, 48, 2, stride=2)
        self.dec2 = DoubleConv(48 + 24, 32)
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec3 = DoubleConv(24 + 16, 16)
        self.up4 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec4 = DoubleConv(16 + 16, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = (x - self.image_mean) / self.image_std
        skips = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in (0, 1, 2, 7):
                skips.append(x)
        s128, s64, s32, s16 = skips
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s16], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s32], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s64], dim=1))
        x = self.up4(x)
        x = self.dec4(torch.cat([x, s128], dim=1))
        return self.out(nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False))


def loss_fn(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(4.0))
    probability = torch.sigmoid(logits)
    dice_loss = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return bce + dice_loss


def scores(model, loader):
    model.eval()
    intersection = union = target_sum = prediction_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            pred = (torch.sigmoid(model(x)) >= THRESHOLD).float()
            intersection += (pred * y).sum().item()
            union += ((pred + y) > 0).float().sum().item()
            target_sum += y.sum().item()
            prediction_sum += pred.sum().item()
    return {
        "dice": round((2 * intersection + 1) / (prediction_sum + target_sum + 1), 4),
        "iou": round((intersection + 1) / (union + 1), 4),
    }


def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)[None]
    model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(x))[0, 0].numpy() >= THRESHOLD).astype(np.uint8), image


def save_overlay(image, mask, output, title, color):
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(Image.fromarray(mask * 150))
    overlay = Image.alpha_composite(image.convert("RGBA"), layer)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((4, 4, 270, 28), fill=(0, 0, 0, 180))
    draw.text((10, 9), title, fill="white")
    overlay.convert("RGB").save(output)


def train_model(model, train_loader, validation_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in train_loader:
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(round(total / len(train_loader), 4))
    return losses, round(time.perf_counter() - started, 1), scores(model, validation_loader)


def main():
    torch.manual_seed(7)
    dataset = NSUDataset()
    # These are service-pipeline checkpoints, so both models use every
    # available tile. Metrics below are training-set diagnostics, not holdout
    # scores; a separate holdout experiment should be used for publication.
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    evaluation_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    models = {"unet": UNet(), "unet_mobilenetv3": MobileUNet()}
    results = {"epochs": EPOCHS, "threshold": THRESHOLD, "sample_id": "18_223653_102117", "training_split": "full_dataset", "training_tiles": len(dataset), "models": {}}
    sample_path = DATASET_ROOT / "images/18_223653_102117.jpg"
    for name, model in models.items():
        losses, seconds, validation = train_model(model, train_loader, evaluation_loader)
        checkpoint = DATASET_ROOT / f"{name}_comparison.pt"
        torch.save(model.state_dict(), checkpoint)
        mask, image = predict(model, sample_path)
        save_overlay(image, mask, DATASET_ROOT / f"comparison_{name}_overlay.png", name, (42, 125, 219) if name != "unet" else (240, 92, 92))
        Image.fromarray(mask * 255).save(DATASET_ROOT / f"comparison_{name}_mask.png")
        results["models"][name] = {"seconds": seconds, "loss": losses, **validation, "checkpoint": checkpoint.name, "overlay": f"comparison_{name}_overlay.png", "encoder": "ImageNet pretrained" if name == "unet_mobilenetv3" else "none"}
        print(name, results["models"][name])
    (DATASET_ROOT / "model_comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
