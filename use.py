import torch
import rasterio
import numpy as np
import os
from patchify import patchify, unpatchify
import segmentation_models_pytorch as smp
from datetime import datetime

RAW_PATH = "raw/2021.tif"
OUTPUT_PATH = "generated/2021_landcover.tif"
CHECKPOINT_PATH = "models/model-effnet.pth"
PATCH_SIZE = 256
NUM_CLASSES = 8

original_classes = [0, 1, 2, 5, 7, 8, 10, 11]
remapped_classes = list(range(NUM_CLASSES))
class_mapping = {new: old for new, old in enumerate(original_classes)}

def load_raw_image(path):
    """Load raw satellite image and normalize"""
    with rasterio.open(path) as src:
        image = src.read().transpose(1, 2, 0).astype(np.float32)
        profile = src.profile

    # Clip and normalize
    image = np.clip(image, 0, 10000) / 10000.0
    return image, profile

def pad_image(image, patch_size):
    """Pad image to be divisible by patch_size"""
    h, w, c = image.shape
    h_target = ((h + patch_size - 1) // patch_size) * patch_size
    w_target = ((w + patch_size - 1) // patch_size) * patch_size

    pad_h = h_target - h
    pad_w = w_target - w
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return image_padded, (h, w)

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    image, profile = load_raw_image(RAW_PATH)
    image_padded, (orig_h, orig_w) = pad_image(image, PATCH_SIZE)

    # Patchify
    patches = patchify(
        image_padded, (PATCH_SIZE, PATCH_SIZE, 4), step=PATCH_SIZE
    ).reshape(-1, PATCH_SIZE, PATCH_SIZE, 4)

    # Load model
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=4,
        classes=NUM_CLASSES,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for patch in patches:
            tensor = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
            output = model(tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            preds.append(pred)

    # Reconstruct full mask
    pred_patches = np.array(preds).reshape(
        image_padded.shape[0] // PATCH_SIZE,
        image_padded.shape[1] // PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    full_mask = unpatchify(pred_patches, image_padded.shape[:2])

    # Crop back to original size
    full_mask = full_mask[:orig_h, :orig_w]

    # Save as GeoTIFF with metadata
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw"
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(full_mask.astype(np.uint8), 1)

        # Add metadata
        metadata = {
            "MODEL": "U-Net (EfficientNet-B4 backbone)",
            "NUM_CLASSES": str(NUM_CLASSES),
            "CLASS_MAPPING": str(class_mapping),
            "TRAIN_EPOCH": str(checkpoint.get("epoch", "unknown")),
            "VAL_LOSS": str(checkpoint.get("val_loss", "unknown")),
            "GENERATED": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        dst.update_tags(**metadata)

    print(f"Landcover map saved at {OUTPUT_PATH} with metadata")

if __name__ == "__main__":
    main()
