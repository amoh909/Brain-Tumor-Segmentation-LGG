import torch
import matplotlib.pyplot as plt

def calculate_metrics(pred, target, smooth=1e-6):

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    return dice.item(), iou.item()

def visualize_and_save(image, mask_true, mask_pred, save_path, title_prefix=""):

    # Convert tensors to numpy arrays of shape (H, W)
    image_np = image.squeeze().cpu().numpy()
    mask_true_np = mask_true.squeeze().cpu().numpy()
    mask_pred_np = mask_pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title(f"{title_prefix} - Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask_true_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    axes[2].imshow(mask_pred_np, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def predict_mask(outputs):

    preds = torch.sigmoid(outputs)
    mask_pred = (preds > 0.5).float()
    return mask_pred
