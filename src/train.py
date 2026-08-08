import argparse
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset
from dataset_b2 import UdiatDatasetB2
from losses import BCEDiceLoss, BoundaryAwareDESLLoss, DESLLoss
from metrics import compute_metrics
from splits import build_dataset, build_split, load_split, save_split
from transforms import get_eval_transform, get_train_transform
from aim_unet import AIMUNet
from unet import UNet
from vss_unet import VSSUNet


def build_model(name, return_branch_outputs=False):
    if name == "unet":
        return UNet(in_channels=1, out_channels=1)
    if name == "vss_unet":
        return VSSUNet(in_channels=1, out_channels=1)
    if name == "aim_unet":
        return AIMUNet(in_channels=1, out_channels=1, return_branch_outputs=return_branch_outputs)
    raise ValueError(f"unknown model: {name}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EMA:
    """Exponential moving average of model weights.

    Evaluating/saving the EMA weights instead of the raw end-of-epoch weights
    smooths out the epoch-to-epoch noise seen across every run so far (best
    val_dice often landing on what looks like a lucky single epoch).
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name] = param.clone()

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


def lr_at_epoch(epoch, base_lr, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay to ~0."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=0.0, ema=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "boundary_dice": 0.0}
    n_batches = 0

    with torch.enable_grad() if is_train else torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            output = model(images)
            if isinstance(output, tuple):
                preds, aux = output
                loss = loss_fn(preds, masks, aux)
            else:
                preds = output
                loss = loss_fn(preds, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item()
            batch_metrics = compute_metrics(preds.detach(), masks)
            for key in metric_sums:
                metric_sums[key] += batch_metrics[key]
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_metrics = {key: value / n_batches for key, value in metric_sums.items()}
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", choices=["busi", "b2"], default="busi")
    parser.add_argument("--model", choices=["unet", "vss_unet", "aim_unet"], default="unet")
    parser.add_argument("--loss", choices=["bce_dice", "desl", "boundary_desl"], default="bce_dice")
    parser.add_argument("--lambda_bdry", type=float, default=0.5)
    parser.add_argument("--split_path", default="data/splits/busi_split.json")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()
    if args.dataset == "b2" and args.split_path == "data/splits/busi_split.json":
        args.split_path = "data/splits/b2_split.json"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset_cls = BUSIDataset if args.dataset == "busi" else UdiatDatasetB2

    split_path = Path(args.split_path)
    if split_path.exists():
        split = load_split(split_path)
    else:
        split = build_split(build_dataset(args.dataset, args.data_root), seed=args.seed)
        save_split(split, split_path)

    train_records, val_records = split["train"], split["val"]
    if args.limit_samples:
        train_records = train_records[: args.limit_samples]
        val_records = val_records[: max(1, args.limit_samples // 4)]

    train_dataset = dataset_cls.from_records(
        train_records, transform=get_train_transform(args.image_size)
    )
    val_dataset = dataset_cls.from_records(
        val_records, transform=get_eval_transform(args.image_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    needs_branches = args.loss in ("desl", "boundary_desl")
    model = build_model(args.model, return_branch_outputs=needs_branches).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, args.ema_decay)
    ema_model = build_model(args.model, return_branch_outputs=needs_branches).to(device)
    if args.loss == "boundary_desl":
        loss_fn = BoundaryAwareDESLLoss(lambda_bdry=args.lambda_bdry)
    elif args.loss == "desl":
        loss_fn = DESLLoss()
    else:
        loss_fn = BCEDiceLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"

    fieldnames = [
        "epoch", "train_loss", "val_loss",
        "val_dice", "val_iou", "val_precision", "val_recall", "val_boundary_dice",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fieldnames)

    best_val_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = lr_at_epoch(epoch, args.lr, args.warmup_epochs, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_loss, _ = run_epoch(
            model, train_loader, loss_fn, device, optimizer, grad_clip=args.grad_clip, ema=ema
        )

        ema.apply_to(ema_model)
        val_loss, val_metrics = run_epoch(ema_model, val_loader, loss_fn, device, optimizer=None)

        print(
            f"epoch {epoch}/{args.epochs} - lr {lr:.6f} - train_loss {train_loss:.4f} - "
            f"val_loss {val_loss:.4f} - val_dice {val_metrics['dice']:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                val_metrics["boundary_dice"],
            ])

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(ema_model.state_dict(), checkpoint_dir / "best_model.pth")

    print(f"best val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
