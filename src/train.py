import os
import sys
import torch
import json

sys.path.append(os.getcwd())
import config
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LGGSegmentationDataset
from src.augmentations import get_train_transforms, get_val_transforms
from src.model import get_model
from src.losses import get_loss_function

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
    train_csv = config.TRAIN_CSV
    validate_csv = config.VAL_CSV
    checkpoints_dir = config.CHECKPOINTS_DIR
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
        shuffle = True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        num_workers=0
    )

    model = get_model(config.MODEL_TYPE, in_channels=1, out_channels=1).to(device)
    loss_fn = get_loss_function(config.LOSS_TYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
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

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        # Save best model, assume we overfit, so we still have a checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoints_dir, f"best_model_{config.EXPERIMENT_ID}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("\nTraining complete.")
    print("Train losses:", train_losses)
    print("Val losses:", val_losses)

    history = {
        "experiment_id": config.EXPERIMENT_ID,
        "description": f"{config.MODEL_TYPE}, loss={config.LOSS_TYPE}, lr={config.LEARNING_RATE}",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss
    }

    history_path = os.path.join(checkpoints_dir, f"history_{config.EXPERIMENT_ID}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()