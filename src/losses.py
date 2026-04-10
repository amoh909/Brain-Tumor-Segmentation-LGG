import torch
import torch.nn as nn 

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-6): 
        super().__init__()
        self.smooth = smooth # Smoothing factor to avoid dividing by 0

    def forward(self, pred, target):
        """
        Computes the Dice Loss for binary segmentation.
        Args:
            pred: model predictions predictions of size (batch_size, 1, H, W).
            target: ground truth values of size (batch_size, 1, H, W).
        Returns:
            Scalar Dice Loss.
        """
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum(dim = (2, 3)) ## True Positives (TP)
        union = pred.sum(dim = (2, 3)) + target.sum(dim = (2, 3)) ## (TP + FP) + (TP + FN)

        dice = (2.0 * intersection + self.smooth)/(union + self.smooth)

        return 1.0 - dice.mean()
    
class BCEDiceLoss(nn.Module): ## We use both BCEWithLogitsLoss and DiceLoss as a joint loss function, note that we did not apply sigmoid in the model since BCEWithLogitsLoss applies it internally
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return bce_loss + dice_loss