"""Run a visible inference demo on one real Namseoul University tile."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch

from train_unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="nsu_real_dataset/unet_nsu_walkable.pt")
    parser.add_argument("--image", default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    root = Path(__file__).parent
    image_path = Path(args.image) if args.image else root / "nsu_real_dataset/images/18_223653_102117.jpg"
    model = UNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)[None]
    with torch.no_grad():
        probability = torch.sigmoid(model(x))[0, 0].numpy()
    mask = (probability >= args.threshold).astype(np.uint8)
    red = Image.new("RGBA", image.size, (255, 40, 40, 0))
    red.putalpha(Image.fromarray(mask * 150))
    overlay = Image.alpha_composite(image.convert("RGBA"), red)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((4, 4, 220, 28), fill=(0, 0, 0, 180))
    draw.text((10, 9), f"NSU predicted walkable: {image_path.name}", fill="white")
    out = root / "nsu_real_dataset" / "demo_overlay.png"
    overlay.convert("RGB").save(out)
    Image.fromarray(mask * 255).save(root / "nsu_real_dataset" / "demo_predicted_mask.png")
    print(f"image={image_path}")
    print(f"overlay={out}")
    print(f"positive_pixels={int(mask.sum())}")


if __name__ == "__main__":
    main()
