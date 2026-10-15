from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255


class LoveDADataset(Dataset):
    def __init__(self, root, split="Train", crop_size=512, augment=False):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.samples = []

        for domain in ["Urban", "Rural"]:
            img_dir = self.root / split / domain / "images"
            if not img_dir.exists():
                img_dir = self.root / split / domain / "images_png"

            lbl_dir = self.root / split / domain / "masks"
            if not lbl_dir.exists():
                lbl_dir = self.root / split / domain / "masks_png"
            if not lbl_dir.exists():
                lbl_dir = self.root / split / domain / "labels"

            if not img_dir.exists() or not lbl_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory for {split}/{domain}: "
                    f"images={img_dir}, labels={lbl_dir}"
                )

            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    self.samples.append((p, lbl_dir, domain))

        n_urban = sum(1 for _, _, d in self.samples if d == "Urban")
        n_rural = sum(1 for _, _, d in self.samples if d == "Rural")
        print(f"[{split}] Loaded {len(self.samples)} images "
              f"({n_urban} Urban, {n_rural} Rural)")

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split={split}")

        self.validate_samples()

        self.train_tf = A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.val_tf = A.Compose([
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def validate_samples(self):
        errors = []
        for img_path, lbl_dir, domain in self.samples:
            mask_path = None
            for ext in [".png", ".tif", ".tiff"]:
                cand = lbl_dir / (img_path.stem + ext)
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                errors.append(f"Missing mask: {domain}/{img_path.name}")
                continue
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                errors.append(f"Unreadable: {domain}/{img_path.name}")
                continue
            if image.shape[:2] != mask.shape[:2]:
                errors.append(
                    f"Shape mismatch: {domain}/{img_path.name} "
                    f"{image.shape[:2]} vs {mask.shape[:2]}"
                )
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            unique = np.unique(mask)
            if unique.min() < 0 or unique.max() > 7:
                errors.append(
                    f"Invalid mask values in {domain}/{img_path.name}: "
                    f"min={unique.min()}, max={unique.max()}"
                )
        if errors:
            preview = "\n".join(errors[:20])
            raise RuntimeError(
                f"Dataset validation failed ({len(errors)} errors):\n{preview}"
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Unable to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.int64)
        mapped = np.full_like(mask, IGNORE_INDEX, dtype=np.int64)
        valid = (mask >= 1) & (mask <= 7)
        mapped[valid] = mask[valid] - 1
        return mapped

    def __getitem__(self, idx):
        img_path, lbl_dir, domain = self.samples[idx]
        mask_path = None
        for ext in [".png", ".tif", ".tiff"]:
            cand = lbl_dir / (img_path.stem + ext)
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_path.name}")

        image = self._read_image(img_path)
        mask = self._read_mask(mask_path)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch: {img_path.name} "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        h, w = image.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            scale = max(self.crop_size / h, self.crop_size / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        tf = self.train_tf if self.augment else self.val_tf
        out = tf(image=image, mask=mask)
        return {"image": out["image"], "mask": out["mask"].long(),
                "name": f"{domain}/{img_path.name}"}