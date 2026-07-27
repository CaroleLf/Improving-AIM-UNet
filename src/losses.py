import torch
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


class DESLLoss(nn.Module):
    """BCE + Dice + diversity regularizer over AIM branch embeddings (paper eq. 9-11)."""

    def __init__(self, lambda_bce=0.5, lambda_div=0.1, lambda_cos=0.1, lambda_var=0.05,
                 eps=1e-6, nu=0.5):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_div = lambda_div
        self.lambda_cos = lambda_cos
        self.lambda_var = lambda_var
        self.eps = eps
        self.nu = nu
        self.bce = nn.BCELoss()

    def _diversity_loss(self, branch_outputs):
        pooled = [b.mean(dim=(2, 3)) for b in branch_outputs]  # globally pooled f_i, (B, C) each
        M = len(pooled)

        cos_sum = 0.0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                fi, fj = pooled[i], pooled[j]
                sim = (fi * fj).sum(dim=1) / ((fi.norm(dim=1) + self.eps) * (fj.norm(dim=1) + self.eps))
                cos_sum = cos_sum + sim.mean()
        # NOTE: paper's eq. 10 has a leading minus sign on this term, but added
        # positively to a loss being minimized (eq. 9), that would reward high
        # cosine similarity instead of penalizing it, contradicting the paper's
        # own stated intent ("increases angular dispersion"). Sign dropped here
        # so minimizing this term actually pushes branches toward orthogonality.
        l_cos = cos_sum / (M * (M - 1))

        l_var = 0.0
        for i in range(M):
            l_var = l_var + torch.clamp(self.nu - pooled[i].std(dim=1).mean(), min=0.0)
        l_var = l_var / M

        return self.lambda_cos * l_cos + self.lambda_var * l_var

    def forward(self, pred, target, aim_branches):
        bce_loss = self.bce(pred, target)
        dice_loss = soft_dice_loss(pred, target)

        div_loss = 0.0
        for branches in aim_branches:
            div_loss = div_loss + self._diversity_loss(branches)
        div_loss = div_loss / len(aim_branches)

        return (self.lambda_bce * bce_loss
                + (1.0 - self.lambda_bce) * dice_loss
                + self.lambda_div * div_loss)
