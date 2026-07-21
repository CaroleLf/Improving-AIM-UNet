import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset
from losses import BCEDiceLoss
from metrics import compute_metrics
from splits import build_split, load_split, save_split
from transforms import get_eval_transform, get_train_transform
from aim_unet import AIMUNet
from unet import UNet
from vss_unet import VSSUNet


def build_model(name):
    if name == "unet":
        return UNet(in_channels=1, out_channels=1)
    if name == "vss_unet":
        return VSSUNet(in_channels=1, out_channels=1)
    if name == "aim_unet":
        return AIMUNet(in_channels=1, out_channels=1)
    raise ValueError(f"unknown model: {name}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n_batches = 0

    with torch.enable_grad() if is_train else torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
    parser.add_argument("--model", choices=["unet", "vss_unet", "aim_unet"], default="unet")
    parser.add_argument("--split_path", default="data/splits/busi_split.json")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    split_path = Path(args.split_path)
    if split_path.exists():
        split = load_split(split_path)
    else:
        split = build_split(args.data_root, seed=args.seed)
        save_split(split, split_path)

    train_records, val_records = split["train"], split["val"]
    if args.limit_samples:
        train_records = train_records[: args.limit_samples]
        val_records = val_records[: max(1, args.limit_samples // 4)]

    train_dataset = BUSIDataset.from_records(
        train_records, transform=get_train_transform(args.image_size)
    )
    val_dataset = BUSIDataset.from_records(
        val_records, transform=get_eval_transform(args.image_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEDiceLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"

    fieldnames = [
        "epoch", "train_loss", "val_loss",
        "val_dice", "val_iou", "val_precision", "val_recall",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fieldnames)

    best_val_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, loss_fn, device, optimizer=None)

        print(
            f"epoch {epoch}/{args.epochs} - train_loss {train_loss:.4f} - "
            f"val_loss {val_loss:.4f} - val_dice {val_metrics['dice']:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
            ])

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")

    print(f"best val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
