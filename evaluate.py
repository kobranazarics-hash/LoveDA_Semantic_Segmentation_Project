import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LoveDADataset, NUM_CLASSES, IGNORE_INDEX, CLASSES
from model import build_model
from utils import update_confusion_matrix, compute_metrics_from_confusion


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LoveDADataset(
        args.data_root,
        split="Val",
        crop_size=args.crop_size,
        augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    model = build_model(NUM_CLASSES, pretrained=False).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)["out"]
            preds = logits.argmax(dim=1)

            update_confusion_matrix(
                confusion, masks, preds,
                NUM_CLASSES, IGNORE_INDEX
            )

    metrics = compute_metrics_from_confusion(confusion)

    print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Overall mIoU:      {metrics['miou'] * 100:.2f}%")
    print(f"Overall Dice:      {metrics['dice'] * 100:.2f}%")
    print("\nClass-wise IoU:")
    for name, value in zip(CLASSES, metrics["iou_per_class"]):
        print(f"{name:12s}: {value * 100:.2f}%")

    print("\nClass-wise Dice:")
    for name, value in zip(CLASSES, metrics["dice_per_class"]):
        print(f"{name:12s}: {value * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args())
import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import LoveDADataset, NUM_CLASSES, IGNORE_INDEX, CLASSES
from model import build_model
from utils import update_confusion_matrix, compute_metrics_from_confusion


CLASS_COLORS = np.array([
    [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
    [159, 129, 183], [0, 255, 0], [255, 195, 128],
], dtype=np.uint8)


class FullImageValDataset(Dataset):
    def __init__(self, root, split="Val"):
        self.root = Path(root)
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
                raise FileNotFoundError(f"Missing dirs for {split}/{domain}")
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    self.samples.append((p, lbl_dir, domain))
        n_urban = sum(1 for _, _, d in self.samples if d == "Urban")
        n_rural = sum(1 for _, _, d in self.samples if d == "Rural")
        print(f"[FullImageVal] {len(self.samples)} images "
              f"({n_urban} Urban, {n_rural} Rural)")
        self.tf = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),
                             ToTensorV2()])

    def __len__(self):
        return len(self.samples)

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
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None: raise RuntimeError(f"Cannot read: {img_path}")
        if mask is None: raise RuntimeError(f"Cannot read: {mask_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mask.ndim == 3: mask = mask[..., 0]
        if mask.shape[:2] != img.shape[:2]:
            raise ValueError(f"Shape mismatch: {img_path.name}")
        mapped = np.full_like(mask, IGNORE_INDEX, dtype=np.int64)
        valid = (mask >= 1) & (mask <= 7)
        mapped[valid] = mask[valid] - 1
        out = self.tf(image=img, mask=mapped)
        return {"image": out["image"], "mask": out["mask"].long(),
                "name": f"{domain}/{img_path.name}"}


def colorize(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(CLASS_COLORS)):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb


def save_prediction(image, mask, pred, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[0].imshow(img); axes[0].set_title("Input Image")
    axes[1].imshow(colorize(mask.cpu().numpy())); axes[1].set_title("Ground Truth")
    axes[2].imshow(colorize(pred.cpu().numpy())); axes[2].set_title("Prediction")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


@torch.no_grad()
def sliding_window_inference(model, image, num_classes=7, window=512, stride=256):
    B, C, H, W = image.shape
    pad_h = max(0, window - H)
    pad_w = max(0, window - W)
    if pad_h > 0 or pad_w > 0:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        H, W = image.shape[2], image.shape[3]
    output = torch.zeros(B, num_classes, H, W, device=image.device)
    count = torch.zeros(B, 1, H, W, device=image.device)
    ys = list(range(0, H - window + 1, stride))
    xs = list(range(0, W - window + 1, stride))
    if ys and ys[-1] != H - window: ys.append(H - window)
    if xs and xs[-1] != W - window: xs.append(W - window)
    for y in ys:
        for x in xs:
            patch = image[:, :, y:y + window, x:x + window]
            logits = model(patch)["out"]
            output[:, :, y:y + window, x:x + window] += logits
            count[:, :, y:y + window, x:x + window] += 1
    output = output / count.clamp(min=1)
    if pad_h > 0 or pad_w > 0:
        output = output[:, :, :H - pad_h, :W - pad_w]
    return output


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    if args.mode == "sliding":
        dataset = FullImageValDataset(args.data_root, split="Val")
        print("Mode: sliding-window on full images")
        batch_size = 1
    else:
        dataset = LoveDADataset(args.data_root, split="Val",
                                crop_size=args.crop_size, augment=False)
        print(f"Mode: center-crop {args.crop_size}x{args.crop_size}")
        batch_size = args.batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=args.workers)
    model = build_model(NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if args.mode == "sliding":
                logits = sliding_window_inference(model, images)
            else:
                logits = model(images)["out"]
            preds = logits.argmax(dim=1)
            update_confusion_matrix(confusion, masks, preds, NUM_CLASSES, IGNORE_INDEX)
            if idx < 3:
                save_prediction(images[0], masks[0], preds[0],
                                f"results/sample_{idx}.png")
    metrics = compute_metrics_from_confusion(confusion)
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Overall mIoU:     {metrics['miou']*100:.2f}%")
    print(f"Overall Dice:     {metrics['dice']*100:.2f}%")
    print("\nClass-wise IoU:")
    for name, value in zip(CLASSES, metrics["iou_per_class"]):
        print(f"{name:12s}: {value*100:.2f}%")
    print("\nClass-wise Dice:")
    for name, value in zip(CLASSES, metrics["dice_per_class"]):
        print(f"{name:12s}: {value*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["center", "sliding"], default="center")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args())