import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset
from dataset_b2 import UdiatDatasetB2
from losses import BCEDiceLoss, BoundaryAwareDESLLoss, DESLLoss
from splits import load_split
from train import build_model, run_epoch
from transforms import get_eval_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", choices=["busi", "b2"], default="busi")
    parser.add_argument("--model", choices=["unet", "vss_unet", "aim_unet"], default="unet")
    parser.add_argument("--loss", choices=["bce_dice", "desl", "boundary_desl"], default="bce_dice")
    parser.add_argument("--lambda_bdry", type=float, default=0.5)
    parser.add_argument("--split_path", default="data/splits/busi_split.json")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    if args.dataset == "b2" and args.split_path == "data/splits/busi_split.json":
        args.split_path = "data/splits/b2_split.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cls = BUSIDataset if args.dataset == "busi" else UdiatDatasetB2
    split = load_split(Path(args.split_path))
    test_dataset = dataset_cls.from_records(
        split["test"], transform=get_eval_transform(args.image_size)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    needs_branches = args.loss in ("desl", "boundary_desl")
    model = build_model(args.model, return_branch_outputs=needs_branches).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.loss == "boundary_desl":
        loss_fn = BoundaryAwareDESLLoss(lambda_bdry=args.lambda_bdry)
    elif args.loss == "desl":
        loss_fn = DESLLoss()
    else:
        loss_fn = BCEDiceLoss()
    test_loss, test_metrics = run_epoch(model, test_loader, loss_fn, device, optimizer=None)

    print(f"test_loss: {test_loss:.4f}")
    for name, value in test_metrics.items():
        print(f"test_{name}: {value:.4f}")


if __name__ == "__main__":
    main()
