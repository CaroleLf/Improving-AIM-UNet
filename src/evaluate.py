import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset
from losses import BCEDiceLoss
from splits import load_split
from train import run_epoch
from transforms import get_eval_transform
from unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_path", default="data/splits/busi_split.json")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_split(Path(args.split_path))
    test_dataset = BUSIDataset.from_records(
        split["test"], transform=get_eval_transform(args.image_size)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    loss_fn = BCEDiceLoss()
    test_loss, test_metrics = run_epoch(model, test_loader, loss_fn, device, optimizer=None)

    print(f"test_loss: {test_loss:.4f}")
    for name, value in test_metrics.items():
        print(f"test_{name}: {value:.4f}")


if __name__ == "__main__":
    main()
