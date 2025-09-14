import rasterio
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from patchify import patchify
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

RAW_PATH = "raw/2021.tif"
PATCH_SIZE = 256
BATCH_SIZE = 12
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


os.makedirs("raw", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_and_preprocess(image_path):
    """Load raw satellite image and normalize"""
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0).astype(np.float32)

        print(f"Raw image stats - Min: {np.min(image)}, Max: {np.max(image)}, NaN count: {np.sum(np.isnan(image))}")

        # Replace NaN with 0
        if np.any(np.isnan(image)):
            print("Warning: Image contains NaN values. Replacing with 0.")
            image = np.nan_to_num(image, nan=0.0)

        # Normalize reflectance values
        image = np.clip(image, 0, 10000) / 10000.0

    h, w, c = image.shape
    h_target = ((h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    w_target = ((w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    pad_h, pad_w = h_target - h, w_target - w
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    print(f"Final image stats - Min: {np.min(image_padded)}, Max: {np.max(image_padded)}")
    return image_padded


class RawDataset(Dataset):
    """Dataset for autoencoder training (input = target = raw patch)"""
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        tensor = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32)
        return tensor, tensor  # input = target


def main():
    # Load raw image
    image = load_and_preprocess(RAW_PATH)

    # Extract patches
    patches = patchify(image, (PATCH_SIZE, PATCH_SIZE, image.shape[2]), step=PATCH_SIZE)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, image.shape[2])

 
    X_train, X_val = train_test_split(patches, test_size=0.2, random_state=42)

   
    train_dataset = RawDataset(X_train)
    val_dataset = RawDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    # Model (autoencoder-style UNet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = image.shape[2]
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,        # we will load our own checkpoint
        in_channels=in_channels,
        classes=in_channels,         # autoencoder: output = input
    ).to(device)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load supervised checkpoint (encoder + decoder only)
    checkpoint_path = "models/model-effnet.pth"
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_path):
        print("Loading encoder+decoder weights from supervised checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        # Drop segmentation head (doesn't match autoencoder output)
        for key in list(state_dict.keys()):
            if "segmentation_head" in key:
                del state_dict[key]

        # Load rest of weights
        model.load_state_dict(state_dict, strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, checkpoint_path)

        print(
            f"Epoch {epoch+1}/{start_epoch + NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )


if __name__ == "__main__":
    main()
