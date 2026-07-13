from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    def __init__(self, root_dir, class_names=("benign", "malignant"), transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for class_name in class_names:
            class_dir = self.root_dir / class_name
            image_paths = sorted(
                p for p in class_dir.glob("*.png") if "_mask" not in p.stem
            )
            if not image_paths:
                raise FileNotFoundError(
                    f"No images found in {class_dir} (expected files not containing '_mask')."
                )

            for image_path in image_paths:
                mask_paths = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
                if not mask_paths:
                    raise FileNotFoundError(
                        f"No mask found for {image_path} (expected '{image_path.stem}_mask*.png')."
                    )
                self.samples.append((image_path, mask_paths, class_name))

    @classmethod
    def from_records(cls, records, transform=None):
        obj = cls.__new__(cls)
        obj.root_dir = None
        obj.transform = transform
        obj.samples = [
            (Path(r["image"]), [Path(m) for m in r["masks"]], r["class"])
            for r in records
        ]
        return obj

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_paths, _ = self.samples[idx]

        image = np.array(Image.open(image_path).convert("L"))
        mask = np.zeros_like(image, dtype=np.uint8)
        for mask_path in mask_paths:
            m = np.array(Image.open(mask_path).convert("L"))
            mask = np.maximum(mask, (m > 0).astype(np.uint8))

        if self.transform is not None:
            augmented = self.transform(image=image[..., None], mask=mask)
            image_t, mask_t = augmented["image"], augmented["mask"].float()
            if mask_t.dim() == 2:
                mask_t = mask_t.unsqueeze(0)
            return image_t, mask_t

        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask


if __name__ == "__main__":
    import sys

    dataset = BUSIDataset(sys.argv[1])
    print(f"{len(dataset)} samples")
    image, mask = dataset[0]
    print("image", image.shape, image.dtype, image.min().item(), image.max().item())
    print("mask", mask.shape, mask.dtype, mask.unique())
