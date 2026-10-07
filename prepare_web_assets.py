from pathlib import Path
from PIL import Image
import numpy as np

root = Path(__file__).parent
for p in (root / "nsu_real_dataset/masks").glob("*.pgm"):
    out = p.with_suffix(".png")
    if not out.exists(): Image.open(p).save(out)

image = Image.open(root / "nsu_real_dataset/images/18_223653_102117.jpg").convert("RGBA")
mask = np.asarray(Image.open(root / "nsu_real_dataset/masks/18_223653_102117.pgm").convert("L"))
yellow = Image.new("RGBA", image.size, (255, 205, 40, 0))
yellow.putalpha(Image.fromarray((mask > 0).astype("uint8") * 175))
Image.alpha_composite(image, yellow).convert("RGB").save(root / "nsu_real_dataset/label_overlay.png")
