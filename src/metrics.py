from losses import extract_boundary


def compute_metrics(pred, target, threshold=0.5, eps=1e-6, boundary_kernel=5):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    boundary_mask = extract_boundary(target, boundary_kernel)
    pred_b = pred_bin * boundary_mask
    target_b = target * boundary_mask
    tp_b = (pred_b * target_b).sum(dim=(1, 2, 3))
    fp_b = (pred_b * (1 - target_b)).sum(dim=(1, 2, 3))
    fn_b = ((boundary_mask - pred_b) * target_b).sum(dim=(1, 2, 3))
    boundary_dice = (2 * tp_b + eps) / (2 * tp_b + fp_b + fn_b + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "boundary_dice": boundary_dice.mean().item(),
    }
