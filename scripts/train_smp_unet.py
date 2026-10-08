"""Train a segmentation_models_pytorch U-Net on the NSU tile dataset."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1] / "data/datasets/namseoul_university"
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class NSUDataset(Dataset):
    def __init__(self):
        self.masks = sorted((ROOT / "masks").glob("*.pgm"))
        self.images = [ROOT / "images" / (p.stem + ".jpg") for p in self.masks]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i):
        image = np.asarray(Image.open(self.images[i]).convert("RGB"), dtype=np.float32) / 255.0
        image = (image - IMAGE_MEAN) / IMAGE_STD
        mask = np.asarray(Image.open(self.masks[i]).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None])

def dice_loss(probability, target):
    smooth = 1.0
    intersection = (probability * target).sum()
    return 1 - (2 * intersection + smooth) / (probability.sum() + target.sum() + smooth)

def main():
    dataset = NSUDataset()
    if not dataset:
        raise SystemExit("데이터셋이 비어 있습니다. 먼저 scripts/build_nsu_dataset.py를 실행하세요.")
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))
    started = time.perf_counter()
    losses = []
    model.train()
    for epoch in range(1, 11):
        total = 0.0
        for images, masks in loader:
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(torch.sigmoid(logits), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        epoch_loss = total / len(loader)
        losses.append(round(epoch_loss, 4))
        print(f"epoch={epoch:02d} loss={epoch_loss:.4f}")
    elapsed = time.perf_counter() - started
    checkpoint = ROOT / "smp_unet_resnet18_imagenet.pt"
    torch.save(model.state_dict(), checkpoint)
    metrics = {
        "model": "segmentation_models_pytorch.Unet",
        "encoder": "resnet18",
        "encoder_weights": "imagenet",
        "epochs": 10,
        "training_tiles": len(dataset),
        "seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / 10, 1),
        "loss": losses,
    }
    model.eval()
    dice_scores, iou_scores = [], []
    sample_id = "18_223653_102117"
    with torch.no_grad():
        for index in range(len(dataset)):
            image_tensor, target = dataset[index]
            logits = model(image_tensor.unsqueeze(0))
            probability = torch.sigmoid(logits)[0, 0].numpy()
            predicted = probability > 0.3
            truth = target[0].numpy() > 0.5
            intersection = np.logical_and(predicted, truth).sum()
            dice_scores.append((2 * intersection) / (predicted.sum() + truth.sum() + 1e-8))
            iou_scores.append(intersection / (np.logical_or(predicted, truth).sum() + 1e-8))
            if dataset.images[index].stem == sample_id:
                base = np.asarray(Image.open(dataset.images[index]).convert("RGB"))
                overlay = base.copy()
                overlay[predicted] = (0.65 * overlay[predicted] + 0.35 * np.array([35, 190, 130])).astype(np.uint8)
                Image.fromarray(overlay).save(ROOT / "smp_unet_resnet18_overlay.png")
    metrics.update({"dice": round(float(np.mean(dice_scores)), 4), "iou": round(float(np.mean(iou_scores)), 4), "threshold": 0.3, "sample_id": sample_id})
    (ROOT / "smp_unet_resnet18_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint}")
    print(f"training_seconds={elapsed:.1f} seconds_per_epoch={elapsed / 10:.1f}")

if __name__ == "__main__":
    main()
