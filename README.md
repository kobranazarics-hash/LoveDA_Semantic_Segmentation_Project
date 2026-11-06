# LoveDA Semantic Segmentation

DeepLabV3 + ResNet-50 on a LoveDA subset (Urban + Rural).
## Dataset Layout

The dataset should be organized as follows:

- `LoveDA/Train/Urban/images/` and `LoveDA/Train/Urban/masks/`
- `LoveDA/Train/Rural/images/` and `LoveDA/Train/Rural/masks/`
- `LoveDA/Val/Urban/images/` and `LoveDA/Val/Urban/masks/`
- `LoveDA/Val/Rural/images/` and `LoveDA/Val/Rural/masks/`

Also accepted folder names: `images_png`, `masks_png`, and `labels`.

Also accepted folder names: images_png, masks_png, and labels.

Labels: raw 0 = ignore, 1–7 = classes. Internal labels: 0–6 + 255.

Expected split: 1,266 training images (633 Urban + 633 Rural) and 840 validation images (420 Urban + 420 Rural).
Training
bash

python train.py --data_root /path/to/LoveDA --experiment baseline --check_counts
python train.py --data_root /path/to/LoveDA --experiment modified --check_counts

The baseline uses Cross-Entropy loss. The modified experiment uses a combined loss:

L = 0.5 × CE + 0.5 × Dice

Loss values are batch-size-weighted averages of batch loss values. CE ignores label 255. Dice uses valid-pixel masking. These are not per-pixel global averages.
Evaluation
bash

python evaluate.py --data_root /path/to/LoveDA --checkpoint checkpoints/best_modified.pth --mode center
python evaluate.py --data_root /path/to/LoveDA --checkpoint checkpoints/best_modified.pth --mode sliding

    Center-crop: central 512 × 512 crop.

    Sliding-window: full-resolution images, using 512 × 512 windows and stride 256.

    The two evaluation protocols cover different spatial regions and are not directly equivalent.

    Both modes use a single entry point in evaluate.py.
Center-Crop and Full-Image Evaluation Results (Modified Model)
Mode	mIoU	Dice
Center crop	47.2%	62.6%
Sliding window	45.8%	60.9%
Published External Reference

Wang et al. (2021):
Metric	Value
mIoU	47.6%
Barren IoU	10.4%
Forest IoU	44.2%

Wang, J., Zheng, Z., Ma, A., Lu, X., & Zhong, Y. (2021). LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2110.08733.

These published results are from DeepLabV3+ with ResNet-50 on the LoveDA test set. They provide external context and are not a directly comparable baseline because the configuration and evaluation protocol differ.
Controlled Comparison (Center-Crop, This Project)
Metric	Baseline	Modified
mIoU	46.8%	47.2%
Dice	61.6%	62.6%
Barren IoU	11.2%	17.5%
Forest IoU	44.0%	44.0%
Water IoU	71.5%	71.5%
Road IoU	50.0%	50.0%
Building IoU	48.5%	48.5%
Agriculture IoU	60.4%	56.9%
Background IoU	42.0%	42.0%

mIoU is the mean of per-class IoU over classes with non-zero union.

Dice is the mean of per-class Dice over the same classes, where:

Dice = 2 × IoU / (1 + IoU)
The modified loss improved Barren IoU from 11.2% to 17.5%, while Agriculture IoU decreased from 60.4% to 56.9%. The effect is class-dependent rather than uniform.
Example Segmentations

The first three validation samples are saved automatically by evaluate.py to:

    results/sample_0.png

    results/sample_1.png

    results/sample_2.png

Their segmentation quality was subsequently inspected visually.

    Image 1 (successful): Building and Agriculture are well segmented; Forest and Water are also correctly identified, with minor boundary noise.

    Image 2 (typical): Building is more fragmented; Forest is over-predicted; Road is less continuous.

    Image 3 (failure): Building is heavily fragmented; Road is largely missing; Background is over-predicted.

These examples illustrate the class-dependent segmentation performance reflected in the quantitative results.
Code Structure

    dataset.py — Urban and Rural loading, mask validation, and augmentation.

    model.py — DeepLabV3 and loss functions.

    train.py — training, validation, and dataset count checks.

    evaluate.py — center-crop and sliding-window evaluation.

    utils.py — metrics, seeding, and visualization.

Initialization

The model is initialized from scratch using:

weights=None, weights_backbone=None

Both the backbone and segmentation head are trained without pretrained weights.
