import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from dataset import BUSIDataset
from dataset_b2 import UdiatDatasetB2


def build_dataset(name, root_dir):
    if name == "busi":
        return BUSIDataset(root_dir)
    if name == "b2":
        return UdiatDatasetB2(root_dir)
    raise ValueError(f"unknown dataset: {name}")


def build_split(dataset, seed=42, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    records = [
        {"image": str(image), "masks": [str(m) for m in masks], "class": cls}
        for image, masks, cls in dataset.samples
    ]
    labels = [r["class"] for r in records]

    train_val, test = train_test_split(
        records, test_size=test_frac, stratify=labels, random_state=seed
    )
    train_val_labels = [r["class"] for r in train_val]
    relative_val = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val_labels, random_state=seed
    )

    return {"train": train, "val": val, "test": test}


def save_split(split, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("--dataset", choices=["busi", "b2"], default="busi")
    parser.add_argument("--out", default="data/splits/busi_split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = build_dataset(args.dataset, args.root_dir)
    split = build_split(dataset, seed=args.seed)
    save_split(split, args.out)
    for name, records in split.items():
        classes = [r["class"] for r in records]
        n_benign = classes.count("benign")
        n_malignant = classes.count("malignant")
        print(f"{name}: {len(records)} samples (benign={n_benign}, malignant={n_malignant})")
