import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

NUM_CLASSES = 7
IGNORE_INDEX = 255


def build_model(num_classes=NUM_CLASSES):
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=False,
    )
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model


class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0
        one_hot = F.one_hot(safe_target.clamp(0, self.num_classes - 1),
                            num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)

    def forward(self, logits, target):
        return 0.5 * self.ce(logits, target) + 0.5 * self.dice(logits, target)