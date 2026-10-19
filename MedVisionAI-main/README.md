# RetinaReach AI — SIH26038

**Explainable AI for Diabetic Retinopathy Screening in Rural India**

This version includes:
- FastAPI backend + SQLite local storage
- Browser-based fundus-camera/webcam capture
- JPG/PNG/WEBP upload
- Image-quality gate
- Four DR/No_DR classifiers: EfficientNet-B3, ResNet50, DenseNet121, MobileNetV3-Large
- Ensemble inference when trained weights are available
- Classical-CV fallback so the app still opens before training
- Screening history and PDF report
- Offline-capable local workflow

## A. Run the website (Windows)

Open **Command Prompt**:

```bat
cd C:\Users\Jenani\Downloads\retinareach-ai
venv\Scripts\activate
cd retinareach-ai\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Keep the terminal running.

### If `venv` does not exist

```bat
cd C:\Users\Jenani\Downloads\retinareach-ai
python -m venv venv
venv\Scripts\activate
cd retinareach-ai\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## B. Train the four models

The supplied dataset is already placed at:

```text
backend\data\dataset\Diagnosis of Diabetic Retinopathy
```

First install training packages:

```bat
cd C:\Users\Jenani\Downloads\retinareach-ai\retinareach-ai\backend
pip install -r train\requirements-training.txt
```

Then test a short run:

```bat
python train\train_four_models.py --epochs 3 --batch-size 8 --workers 0
```

After confirming it works, train longer:

```bat
python train\train_four_models.py --epochs 10 --batch-size 8 --workers 0
```

Optional ImageNet transfer learning:

```bat
python train\train_four_models.py --epochs 5 --batch-size 8 --workers 0 --pretrained
```

Model files will appear in:

```text
backend\models\
```

Restart the FastAPI server after training. The backend will automatically use every compatible `.pth` model it finds.

## C. Fundus camera

On the Screening page choose **Use Fundus Camera / Webcam**. The browser asks for camera permission, shows the live preview, and captures a JPG frame for the same analysis API.

For direct capture, the fundus camera must be exposed to Windows as a normal webcam/UVC device. If the camera only works through proprietary software, export a JPG/PNG from that software and choose **Upload Image**.

## D. Dataset limitation

The supplied dataset contains only two classes: `DR` and `No_DR`. Therefore the four trained models are four different architectures for the same binary classification task. It does **not** provide lesion bounding boxes or segmentation masks, so this project does not falsely claim YOLO lesion detection or U-Net segmentation.

## E. Medical/demo safety

This is an AI-assisted screening prototype, not a clinical diagnostic system. Report metrics from the held-out test set and do not claim 100% accuracy or clinical validity without independent clinical validation.
