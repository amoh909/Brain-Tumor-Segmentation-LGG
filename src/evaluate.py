import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

sys.path.append(os.getcwd())
import src.config as config
from src.dataset import LGGSegmentationDataset
from src.augmentations import get_val_transforms
from src.model import get_model
from src.losses import get_loss_function
from src.utils import calculate_metrics, visualize_and_save, predict_mask


def main():
    evaluation_dir = os.path.join(config.EVALUATION_DIR, config.EXPERIMENT_ID)
    os.makedirs(evaluation_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset and DataLoader
    test_dataset = LGGSegmentationDataset(
        csv_file=config.TEST_CSV,
        transform=get_val_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # Evaluate 1 by 1 for accurate instance tracking
        shuffle=False
    )
    
    # 2. Model Initialization
    model = get_model(config.MODEL_TYPE, in_channels=1, out_channels=1).to(device)
    checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"best_model_{config.EXPERIMENT_ID}.pth")
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}! Using untrained model.")
        
    model.eval()
    loss_fn = get_loss_function(config.LOSS_TYPE)
    
    # 3. Evaluation Loop & Tracking
    results = [] # To store dicts of index, dice, iou
    total_dice = 0.0
    total_iou = 0.0
    total_loss = 0.0
    
    print("Starting evaluation...")
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating", leave=False)
        for i, (images, masks) in enumerate(loop):
            masks = masks.to(device)
            
            # Predict mask (thresholded) and get logits for loss
            images_device = images.to(device)
            outputs = model(images_device)  # logits
            
            # Predict binary masks using the utility
            preds = predict_mask(outputs)
            
            # Calculate classification loss
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            # Calculate metrics
            dice, iou = calculate_metrics(preds, masks)
            total_dice += dice
            total_iou += iou
            
            # Save metrics
            results.append({
                'index': i,
                'image_path': test_dataset.data_info.loc[i, 'image_path'],
                'dice': dice,
                'iou': iou,
            })
            
            loop.set_postfix(dice=f"{dice:.4f}")
            
    num_samples = len(test_loader)
    
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_loss = total_loss / num_samples
    
    # Save aggregate results
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(evaluation_dir, "detailed_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    summary = {
        'total_samples': num_samples,
        'average_loss': avg_loss,
        'average_dice': avg_dice,
        'average_iou': avg_iou
    }
    summary_path = os.path.join(evaluation_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nEvaluation Results Saved to: {evaluation_dir}")
    print(f"Total Samples: {num_samples}")
    print(f"Average Test Loss:  {avg_loss:.4f}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score:  {avg_iou:.4f}\n")
    
    # 4. Best and Worst Predictions
    # Sort results by Dice score ascending
    results.sort(key=lambda x: x['dice'])
    
    n_show = min(5, len(results)) 
    worst_cases = results[:n_show] # Lowest Dice
    best_cases = results[-n_show:] # Highest Dice
    best_cases.reverse()           # Reorder so highest is first
    
    # Process extreme cases dynamically to save memory
    def process_and_visualize(cases, prefix):
        for rank, case in enumerate(cases):
            idx = case['index']
            dice = case['dice']
            
            # Retrieve from dataset directly
            image, mask_true = test_dataset[idx]
            
            # Predict
            with torch.no_grad():
                img_device = image.unsqueeze(0).to(device)
                outputs = model(img_device)
                mask_pred = predict_mask(outputs).cpu().squeeze(0)
                
            save_name = f"{prefix}_{rank+1}_dice_{dice:.4f}.png"
            save_path = os.path.join(evaluation_dir, save_name)
            
            visualize_and_save(
                image, mask_true, mask_pred, 
                save_path, 
                title_prefix=f"{prefix.capitalize()} #{rank+1} (Dice: {dice:.4f})"
            )
            
    if len(results) > 0:
        process_and_visualize(worst_cases, "worst")
        unique_best = [c for c in best_cases if c not in worst_cases]
        if unique_best:
            process_and_visualize(unique_best, "best")

    print(f"Visualizations saved to {evaluation_dir}/")

if __name__ == "__main__":
    main()
