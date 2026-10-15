import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoveDADataset, NUM_CLASSES, IGNORE_INDEX, CLASSES
from model import build_model, ComboLoss
from utils import set_seed, update_confusion_matrix, compute_metrics_from_confusion


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss_weighted = 0.0
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)["out"]
        loss = criterion(logits, masks)
        total_loss_weighted += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        update_confusion_matrix(confusion, masks, preds, NUM_CLASSES, IGNORE_INDEX)
    metrics = compute_metrics_from_confusion(confusion)
    metrics["val_loss"] = total_loss_weighted / max(len(loader.dataset), 1)
    return metrics


def domain_counts(dataset):
    c = Counter(d for _, _, d in dataset.samples)
    return c.get("Urban", 0), c.get("Rural", 0)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = LoveDADataset(args.data_root, split="Train",
                             crop_size=args.crop_size, augment=True)
    val_ds = LoveDADataset(args.data_root, split="Val",
                           crop_size=args.crop_size, augment=False)

    tr_u, tr_r = domain_counts(train_ds)
    va_u, va_r = domain_counts(val_ds)
    print(f"Train: {len(train_ds)} total ({tr_u} Urban, {tr_r} Rural)")
    print(f"Val:   {len(val_ds)} total ({va_u} Urban, {va_r} Rural)")

    if args.check_counts:
        assert len(train_ds) == 1266
        assert len(val_ds) == 840
        assert tr_u == 633 and tr_r == 633
        assert va_u == 420 and va_r == 420
        print("Dataset size check passed.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=torch.cuda.is_available())

    model = build_model(NUM_CLASSES).to(device)

    if args.experiment == "baseline":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    else:
        criterion = ComboLoss(NUM_CLASSES, IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou = -1.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss_weighted = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)["out"]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss_weighted += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        train_loss = running_loss_weighted / max(len(train_ds), 1)
        metrics = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
              f"val_loss={metrics['val_loss']:.4f}, "
              f"mIoU={metrics['miou']:.4f}, Dice={metrics['dice']:.4f}")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": metrics["miou"],
                "dice": metrics["dice"],
                "val_loss": metrics["val_loss"],
                "experiment": args.experiment,
                "classes": CLASSES,
            }, Path(args.output_dir) / f"best_{args.experiment}.pth")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--experiment", choices=["baseline", "modified"], default="modified")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--check_counts", action="store_true")
    args = parser.parse_args()
    train(args)