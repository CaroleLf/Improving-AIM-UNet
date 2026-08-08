import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BUSIDataset
from losses import extract_boundary
from splits import load_split
from train import build_model
from transforms import get_eval_transform


def smallest_lesion_records(records, image_size):
    raw_dataset = BUSIDataset.from_records(records, transform=None)
    areas = []
    for i in range(len(raw_dataset)):
        _, mask = raw_dataset[i]
        areas.append(mask.sum().item())
    order = np.argsort(areas)
    return [records[i] for i in order]


def overlay_gt_boundary(pred_mask, gt_mask, color=(1.0, 0.0, 0.0), kernel_size=3):
    """pred_mask, gt_mask: (H, W) numpy arrays in {0,1}. Returns an (H, W, 3) RGB image."""
    boundary = extract_boundary(
        torch.from_numpy(gt_mask).float().unsqueeze(0).unsqueeze(0), kernel_size=kernel_size
    ).squeeze().numpy()

    rgb = np.stack([pred_mask, pred_mask, pred_mask], axis=-1).astype(np.float32)
    rgb[boundary > 0.5] = color
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_path", default="data/splits/busi_split.json")
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--solutionA_checkpoint", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--select", choices=["smallest", "first"], default="smallest")
    parser.add_argument("--out", default="checkpoints/qualitative_comparison.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_split(Path(args.split_path))
    test_records = split["test"]
    if args.select == "smallest":
        test_records = smallest_lesion_records(test_records, args.image_size)
    test_records = test_records[: args.num_examples]

    dataset = BUSIDataset.from_records(test_records, transform=get_eval_transform(args.image_size))

    baseline = build_model("aim_unet", return_branch_outputs=False).to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))
    baseline.eval()

    solution_a = build_model("aim_unet", return_branch_outputs=False).to(device)
    solution_a.load_state_dict(torch.load(args.solutionA_checkpoint, map_location=device))
    solution_a.eval()

    n = len(dataset)
    col_titles = ["Image", "Ground truth", "Baseline (DESL)", "Solution A"]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for i in range(n):
            image, mask = dataset[i]
            image_batch = image.unsqueeze(0).to(device)

            pred_baseline = (baseline(image_batch).squeeze().cpu() > 0.5).float().numpy()
            pred_solution_a = (solution_a(image_batch).squeeze().cpu() > 0.5).float().numpy()

            image_np = image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()

            panels = [
                np.stack([image_np] * 3, axis=-1),
                np.stack([mask_np] * 3, axis=-1),
                overlay_gt_boundary(pred_baseline, mask_np),
                overlay_gt_boundary(pred_solution_a, mask_np),
            ]

            for j, (panel, title) in enumerate(zip(panels, col_titles)):
                axes[i, j].imshow(panel)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(title)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
