"""Deep-learning inference helpers for the four supported DR classifiers.

Models:
- EfficientNet-B3
- ResNet50
- DenseNet121
- MobileNetV3-Large

Weights are optional. If no trained weights are present, the application
continues to use the classical-CV fallback in retina_analyzer.py.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

MODEL_NAMES = ["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3_large"]
MODEL_LABELS = {
    "efficientnet_b3": "EfficientNet-B3",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


def _torchvision_model(name: str):
    import torch.nn as nn
    from torchvision import models

    if name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def build_model(name: str):
    return _torchvision_model(name)


def _preprocess(image_path: str):
    import torch
    from torchvision import transforms

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def _load_state(model, path: str):
    import torch
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def available_models(models_dir: str) -> List[str]:
    return [n for n in MODEL_NAMES if os.path.isfile(os.path.join(models_dir, f"{n}.pth"))]


def ensemble_predict(image_path: str, models_dir: str) -> Optional[Dict]:
    """Return an ensemble prediction when possible; otherwise use the CV fallback.

    PyTorch and torchvision are optional runtime dependencies. The application must
    remain usable when they are not installed, especially in offline mode.
    """
    names = available_models(models_dir)
    if not names:
        return None

    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        # Deep-learning dependencies are optional. Let retina_analyzer.py use
        # its classical computer-vision fallback instead of crashing screening.
        return None

    x = _preprocess(image_path)
    probabilities = []
    used = []
    for name in names:
        try:
            model = build_model(name)
            _load_state(model, os.path.join(models_dir, f"{name}.pth"))
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            probabilities.append(probs)
            used.append(name)
        except Exception:
            # A single corrupt/incompatible weight file must not break screening.
            continue

    if not probabilities:
        return None

    mean_probs = np.mean(np.stack(probabilities), axis=0)
    # ImageFolder class order is alphabetic: DR=0, No_DR=1.
    dr_probability = float(mean_probs[0])
    no_dr_probability = float(mean_probs[1])
    predicted = "DR" if dr_probability >= no_dr_probability else "No_DR"
    confidence = round(max(dr_probability, no_dr_probability) * 100, 2)

    return {
        "prediction": predicted,
        "dr_probability": round(dr_probability * 100, 2),
        "no_dr_probability": round(no_dr_probability * 100, 2),
        "confidence": confidence,
        "models_used": [MODEL_LABELS[n] for n in used],
    }
