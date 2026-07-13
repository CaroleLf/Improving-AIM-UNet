import torch.nn as nn


def soft_dice_loss(pred, target, eps=1e-6):
    B = pred.size(0)
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, lambda_bce=0.5):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = soft_dice_loss(pred, target)
        return self.lambda_bce * bce_loss + (1.0 - self.lambda_bce) * dice_loss
