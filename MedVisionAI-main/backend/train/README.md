# Training the 4 DR classifiers

This project uses the supplied binary fundus dataset:

- `DR`
- `No_DR`

The four models are trained for the SAME binary classification task:

1. EfficientNet-B3 — main classifier
2. ResNet50 — comparison classifier
3. DenseNet121 — feature-learning classifier
4. MobileNetV3-Large — lightweight classifier

## Dataset layout

```text
backend/data/dataset/Diagnosis of Diabetic Retinopathy/
  train/DR
  train/No_DR
  valid/DR
  valid/No_DR
  test/DR
  test/No_DR
```

## Install training dependencies

From the `backend` folder:

```bat
pip install -r train\requirements-training.txt
```

## Train all four models

Recommended first run on a normal laptop:

```bat
python train\train_four_models.py --epochs 3 --batch-size 8 --workers 0
```

For better training, increase epochs after confirming the first run works:

```bat
python train\train_four_models.py --epochs 10 --batch-size 8 --workers 0
```

Optional ImageNet transfer learning (requires downloading pretrained weights):

```bat
python train\train_four_models.py --epochs 5 --batch-size 8 --workers 0 --pretrained
```

The best validation-F1 checkpoint for each model is saved to:

```text
backend/models/efficientnet_b3.pth
backend/models/resnet50.pth
backend/models/densenet121.pth
backend/models/mobilenet_v3_large.pth
```

The backend automatically detects these files and uses all compatible models as an ensemble. If weights are absent, the app remains runnable using its classical-CV fallback.

## Important

The supplied labels are only DR/No_DR. They do not provide lesion bounding boxes or pixel masks, so this training pipeline does not claim lesion detection or segmentation. Do not claim 100% accuracy or clinical diagnosis without independent clinical validation.
