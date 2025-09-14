import torch
import rasterio
import numpy as np
import os
import segmentation_models_pytorch as smp
from datetime import datetime
from copy import deepcopy

# Config
RAW_PATH = "raw/2019.tif"
OUTPUT_PATH = "generated/2019_landcover.tif"
CHECKPOINT_PATH = "models/model-effnet.pth"
NUM_CLASSES = 8
IGNORE_INDEX = 255

def load_and_pad(image_path):
    """Load raw image and pad to be divisible by 32"""
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0).astype(np.float32)
        meta = deepcopy(src.meta)
        h, w = image.shape[:2]

    # Normalize like in train.py
    image = np.clip(image, 0, 10000) / 10000.0
    divisor = 32
    pad_h = (divisor - (h % divisor)) % divisor
    pad_w = (divisor - (w % divisor)) % divisor

    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    return image_padded, (h, w), meta, pad_h, pad_w

def sliding_window_inference(model, image_tensor, patch_size=512, overlap=32, device="cpu"):
    _, _, H, W = image_tensor.shape
    stride = patch_size - overlap
    pred_mask = np.full((H, W), IGNORE_INDEX, dtype=np.int64)  # prefill with ignore

    model.eval()
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1, x1 = y, x
                y2, x2 = min(y + patch_size, H), min(x + patch_size, W)

                patch = image_tensor[:, :, y1:y2, x1:x2].to(device)
                output = model(patch)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                pred_mask[y1:y2, x1:x2] = pred

    return pred_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load & pad image
    image_padded, (orig_h, orig_w), meta, pad_h, pad_w = load_and_pad(RAW_PATH)

    # Convert to tensor
    image_tensor = (
        torch.tensor(image_padded.transpose(2, 0, 1), dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    # Load model
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=image_padded.shape[-1],
        classes=NUM_CLASSES,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inference
    pred_mask = sliding_window_inference(
        model, image_tensor, patch_size=512, overlap=32, device=device
    )

    # Crop back to original size
    full_mask = pred_mask[:orig_h, :orig_w]

    # ✅ Apply Ignore Index filtering on padded regions
    if pad_h > 0:
        full_mask[-pad_h:, :] = IGNORE_INDEX
    if pad_w > 0:
        full_mask[:, -pad_w:] = IGNORE_INDEX

    # Prepare metadata for output GeoTIFF
    tif_meta = meta.copy()
    tif_meta.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": IGNORE_INDEX,
            "compress": "lzw",
        }
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with rasterio.open(OUTPUT_PATH, "w", **tif_meta) as dst:
        dst.write(full_mask.astype("uint8"), 1)

    print(f"Landcover map saved at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
