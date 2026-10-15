import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_confusion_matrix(confusion, target, prediction, num_classes, ignore_index=255):
    target = target.detach().cpu().numpy().reshape(-1)
    prediction = prediction.detach().cpu().numpy().reshape(-1)
    prediction = np.clip(prediction, 0, num_classes - 1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    target = target[valid]
    prediction = prediction[valid]
    encoded = target * num_classes + prediction
    bincount = np.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bincount.reshape(num_classes, num_classes)
    return confusion


def compute_metrics_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan),
                    where=denom_iou != 0)
    valid = ~np.isnan(iou)
    miou = float(np.mean(iou[valid])) if valid.any() else float("nan")

    denom_dice = 2 * tp + fp + fn
    dice = np.divide(2 * tp, denom_dice, out=np.full_like(tp, np.nan),
                     where=denom_dice != 0)
    mdice = float(np.nanmean(dice))

    total = confusion.sum()
    accuracy = float(tp.sum() / total) if total > 0 else float("nan")

    return {"accuracy": accuracy, "miou": miou, "dice": mdice,
            "iou_per_class": iou, "dice_per_class": dice}


def save_prediction(image, mask, prediction, class_names, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title("Image")
    axes[1].imshow(mask); axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction); axes[2].set_title("Prediction")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)