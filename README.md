# MedVisionAI — SIH26038

## Explainable AI for Diabetic Retinopathy Screening in Rural India

### About the Project
MedVisionAI is an AI-assisted screening prototype designed to support diabetic retinopathy screening in rural and remote areas. It uses deep learning and computer vision to analyze retinal fundus images.

### Features
- FastAPI backend
- SQLite local storage
- Fundus camera / webcam image capture
- JPG, PNG and WEBP image upload
- Image quality checking
- Four deep learning models:
  - EfficientNet-B3
  - ResNet50
  - DenseNet121
  - MobileNetV3-Large
- Ensemble prediction when trained model weights are available
- Classical computer vision fallback
- Screening history and PDF reports
- Offline-capable local workflow

### Technologies Used
- Python
- FastAPI
- SQLite
- PyTorch
- OpenCV
- HTML, CSS and JavaScript

### How to Run the Project

**Step 1:** Install Python on your computer.

**Step 2:** Download or clone this repository.

**Step 3:** Open Command Prompt inside the project folder.

**Step 4:** Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

**Step 5:** Install the required dependencies.

```bash
cd backend
pip install -r requirements.txt
```

**Step 6:** Start the application.

```bash
uvicorn main:app --reload --port 8000
```

**Step 7:** Open the application in your browser.

http://127.0.0.1:8000

### Model Training

Prepare the dataset in the directory expected by the training script.

Install training dependencies:

```bash
pip install -r train/requirements-training.txt
```

Run model training:

```bash
python train/train_four_models.py --epochs 3 --batch-size 8 --workers 0
```

The training configuration and model output paths depend on the project implementation.

### Dataset

APTOS 2019 Blindness Detection Dataset:

https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

The supplied dataset uses the DR and No_DR classes. It does not provide lesion bounding boxes or segmentation masks.

### Project Domain
MedTech / BioTech / HealthTech

### Problem Statement
SIH26038

### Medical Disclaimer
This project is an AI-assisted screening prototype for educational and research purposes. It is not a clinically validated diagnostic system and does not replace professional eye examinations. Model performance must be evaluated on a held-out test set, and clinical use requires independent validation.

### Acknowledgement
Developed as a project focused on improving access to diabetic retinopathy screening in rural India.
