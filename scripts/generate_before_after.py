"""Create same-sample before/after inference assets for the web demo."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_unet import UNet

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/datasets/namseoul_university"
SAMPLE = DATASET / "images/18_223653_102117.jpg"


def predict(model, image):
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)[None]
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(x))[0, 0].numpy()
    return (probability >= 0.3).astype(np.uint8)


def save_overlay(image, mask, output, title, color):
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(Image.fromarray(mask * 150))
    overlay = Image.alpha_composite(image.convert("RGBA"), layer)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((4, 4, 270, 28), fill=(0, 0, 0, 180))
    draw.text((10, 9), title, fill="white")
    overlay.convert("RGB").save(output)


def main():
    torch.manual_seed(7)
    image = Image.open(SAMPLE).convert("RGB")
    before = predict(UNet(), image)
    save_overlay(image, before, DATASET / "demo_before_overlay.png", "Before training · random init", (240, 92, 92))
    Image.fromarray(before * 255).save(DATASET / "demo_before_predicted_mask.png")

    after_model = UNet()
    # Keep SAME SAMPLE aligned with the service-pipeline benchmark checkpoint.
    checkpoint = DATASET / "unet_comparison.pt"
    if not checkpoint.exists():
        checkpoint = DATASET / "unet_nsu_walkable.pt"
    after_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    after = predict(after_model, image)
    save_overlay(image, after, DATASET / "demo_after_overlay.png", "After training · service U-Net", (42, 125, 219))
    Image.fromarray(after * 255).save(DATASET / "demo_after_predicted_mask.png")
    print(f"sample={SAMPLE.name} before_positive_pixels={int(before.sum())} after_positive_pixels={int(after.sum())}")


if __name__ == "__main__":
    main()
