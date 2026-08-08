from pathlib import Path

import openpyxl

from dataset import BUSIDataset


class UdiatDatasetB2(BUSIDataset):
    """Dataset B2 (BUS2017-B / UDIAT): original/GT folders, labels in DatasetB.xlsx.

    Reuses BUSIDataset's __getitem__/__len__/from_records (they only depend on
    self.samples), only the discovery logic in __init__ differs.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = self._discover_samples()

    def _discover_samples(self):
        image_dir = self.root_dir / "original"
        mask_dir = self.root_dir / "GT"
        xlsx_path = self.root_dir / "DatasetB.xlsx"

        labels = self._load_labels(xlsx_path)

        image_paths = sorted(image_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        samples = []
        for image_path in image_paths:
            image_id = image_path.stem
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"No mask found for {image_path} (expected {mask_path})")
            label = labels.get(image_id)
            if label is None:
                raise ValueError(f"No label found in {xlsx_path} for image {image_id}")
            samples.append((image_path, [mask_path], label))
        return samples

    @staticmethod
    def _load_labels(xlsx_path):
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        labels = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            image_id, type_ = row[0], row[1]
            if image_id is None or type_ is None:
                continue
            labels[str(image_id)] = type_.strip().lower()
        return labels


if __name__ == "__main__":
    import sys

    dataset = UdiatDatasetB2(sys.argv[1])
    print(f"{len(dataset)} samples")
    image, mask = dataset[0]
    print("image", image.shape, image.dtype, image.min().item(), image.max().item())
    print("mask", mask.shape, mask.dtype, mask.unique())
    labels = [s[2] for s in dataset.samples]
    print("benign:", labels.count("benign"), "malignant:", labels.count("malignant"))
