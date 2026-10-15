README.md


# LoveDA Semantic Segmentation

DeepLabV3 + ResNet-50 on a LoveDA subset (Urban + Rural).

## Setup

pip install -r requirements.txt

Layout: LoveDA/{Train,Val}/{Urban,Rural}/{images,masks}
Also accepted: images_png, masks_png, labels.
Labels: raw 0 = ignore, 1-7 = classes. Internal: 0-6 + 255.
Expected split: 1266 train (633+633), 840 val (420+420).

## Training

python train.py --data_root /path/to/LoveDA --experiment baseline --check_counts
python train.py --data_root /path/to/LoveDA --experiment modified --check_counts

Losses are batch-size-weighted averages of batch loss values. CE ignores label 255. Dice uses valid-pixel masking from model.py. Not a per-pixel global average.

## Evaluation

python evaluate.py --data_root /path/to/LoveDA --checkpoint checkpoints/best_modified.pth --mode center
python evaluate.py --data_root /path/to/LoveDA --checkpoint checkpoints/best_modified.pth --mode sliding

Center-crop: 512x512 central crop.
Sliding-window: full 1024x1024 image, 512x512 windows, stride 256.
Not directly comparable. Main table uses center-crop.

## Results

Published reference (Wang et al., 2021) — external:

| Metric | Value |
|---|---|
| mIoU | 47.6% |
| Barren IoU | 10.4% |
| Forest IoU | 44.2% |

Wang, J., Zheng, Z., Ma, A., Lu, X., & Zhong, Y. (2021). LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2110.08733.

These published results are from DeepLabV3+ with ResNet-50 on the LoveDA test set. External context, not a directly comparable baseline. Configuration and evaluation protocol differ.

Controlled comparison (center-crop, this project):

| Metric | Baseline | Modified |
|---|---|---|
| mIoU | 46.8% | 47.2% |
| Dice | 61.6% | 62.6% |
| Barren IoU | 11.2% | 17.5% |
| Forest IoU | 44.0% | 44.0% |
| Water IoU | 71.5% | 71.5% |
| Road IoU | 50.0% | 50.0% |
| Building IoU | 48.5% | 48.5% |
| Agriculture IoU | 60.4% | 56.9% |
| Background IoU | 42.0% | 42.0% |

mIoU = mean of per-class IoU over classes with non-zero union.
Dice = 2 * IoU / (1 + IoU), averaged over the same classes.

Barren: 11.2% → 17.5%. Agriculture: 60.4% → 56.9%. Class-dependent effect.

## Example segmentations

evaluate.py saves three examples to results/sample_0.png, sample_1.png, sample_2.png. Selected manually from validation by visual inspection.

- Image 1 (successful): Building and Agriculture well-segmented; Forest and Water correct; minor boundary noise.
- Image 2 (typical): Building more fragmented; Forest over-predicted; Road less continuous.
- Image 3 (failure): Building heavily fragmented; Road largely missing; Background over-predicted.

Consistent with overall mIoU of 47.2% and per-class IoU above.

## Code

- dataset.py — Urban + Rural loading, mask validation, augmentation
- model.py — DeepLabV3 + ComboLoss
- train.py — training, weighted val_loss, dataset count check
- evaluate.py — center + sliding evaluation
- utils.py — metrics, seeding, visualization

## Initialization

weights=None, weights_backbone=None. Backbone and head from scratch.

## Files

- requirements.txt
- training_log.txt
- report.md
- respond.docx