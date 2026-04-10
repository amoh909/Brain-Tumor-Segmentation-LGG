import os
import sys
import torch

sys.path.append(os.getcwd())
import config
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LGGSegmentationDataset
from src.augmentations import get_train_transforms, get_val_transforms
from src.model import UNet
from src.losses import BCEDiceLoss

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    loss = 0.0

    loop = tqdm(dataloader, desc = "Training", leave = False)

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        currentloss = loss_fn(outputs, masks)
        currentloss.backward()
        optimizer.step()
        loss += currentloss.item()
        loop.set_postfix(currentloss=currentloss.item())

    return loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    loss = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc = "Validate", leave = False)

        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            currentloss = loss_fn(outputs, masks)
            loss += currentloss.item()
            loop.set_postfix(currentloss=currentloss.item())

    return loss / len(dataloader)

def main():
    train_csv = "data/processed/train.csv"
    validate_csv = "data/processed/val.csv"
    checkpoints_dir = "outputs/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    train_dataset = LGGSegmentationDataset(
        csv_file = train_csv,
        transform = get_train_transforms()
    )

    val_dataset = LGGSegmentationDataset(
        csv_file = validate_csv,
        transform = get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False
    )

    model = UNet(in_channels=1, out_channels=1).to(device)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # Save best model, assume we overfit, so we still have a checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("\nTraining complete.")
    print("Train losses:", train_losses)
    print("Val losses:", val_losses)


if __name__ == "__main__":
    main()