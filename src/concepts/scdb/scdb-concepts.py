import os
import numpy as np
from PIL import Image

img_dir = "/netscratch/aslam/TCAV/SCDB/concept"
mask_dir = os.path.join(img_dir, "Segmentation")
out_root = "/netscratch/aslam/TCAV/SCDB/concept-dirs"

for mask_name in os.listdir(mask_dir):
    if not mask_name.endswith(".png"):
        continue

    base, concept = mask_name.replace(".png", "").split("_", 1)
    img_name = f"{base}.png"

    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)

    # print(f"{img_path=}")
    # print(f"{mask_path=}")
    # continue

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    # Load image (RGB) and mask (grayscale)
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Sanity check: spatial dimensions must match
    if img.shape[:2] != mask.shape:
        raise ValueError(f"Shape mismatch: {img_name} vs {mask_name}")

    # Enforce binary mask
    mask = (mask > 0).astype(np.uint8)

    # Apply mask (broadcast over channels)
    masked = img * mask[..., None]

    # Output directory per concept
    out_dir = os.path.join(out_root, concept)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, mask_name)
    Image.fromarray(masked).save(out_path)
