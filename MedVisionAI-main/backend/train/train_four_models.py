"""Train four DR/No_DR classifiers on an ImageFolder dataset.

Expected dataset layout:
  backend/data/dataset/Diagnosis of Diabetic Retinopathy/
      train/DR, train/No_DR
      valid/DR, valid/No_DR
      test/DR,  test/No_DR

Usage from backend folder:
  python train/train_four_models.py --epochs 5

Optional ImageNet transfer learning:
  python train/train_four_models.py --epochs 5 --pretrained

The script saves the best validation checkpoint for each model to:
  backend/models/<model_name>.pth
and a summary to backend/models/training_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAMES = ["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3_large"]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name: str, pretrained: bool):
    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    else:
        raise ValueError(name)
    return model


def make_loaders(data_root: Path, batch_size: int, workers: int):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(data_root / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    expected = {"DR": 0, "No_DR": 1}
    if train_ds.class_to_idx != expected or valid_ds.class_to_idx != expected or test_ds.class_to_idx != expected:
        raise RuntimeError(
            "Dataset classes must be exactly DR and No_DR. "
            f"Found train={train_ds.class_to_idx}, valid={valid_ds.class_to_idx}, test={test_ds.class_to_idx}"
        )

    kwargs = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(valid_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
        train_ds.classes,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_y, all_pred, all_prob = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        prob_dr = torch.softmax(logits, dim=1)[:, 0]
        pred = torch.argmax(logits, dim=1)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob_dr.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_pred)
    precision = precision_score(all_y, all_pred, pos_label=0, zero_division=0)
    recall = recall_score(all_y, all_pred, pos_label=0, zero_division=0)
    f1 = f1_score(all_y, all_pred, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score([1 if y == 0 else 0 for y in all_y], all_prob)
    except ValueError:
        auc = float("nan")
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Dataset root containing train/valid/test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for maximum compatibility")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights (internet may be required)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    backend_dir = Path(__file__).resolve().parents[1]
    data_root = Path(args.data).resolve() if args.data else backend_dir / "data" / "dataset" / "Diagnosis of Diabetic Retinopathy"
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not all((data_root / split).is_dir() for split in ("train", "valid", "test")):
        raise FileNotFoundError(f"Dataset folders train/valid/test not found under: {data_root}")

    train_loader, valid_loader, test_loader, classes = make_loaders(data_root, args.batch_size, args.workers)
    print(f"Dataset: {data_root}")
    print(f"Classes: {classes}; train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, test={len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = {"dataset": str(data_root), "device": str(device), "models": {}}

    for name in MODEL_NAMES:
        print(f"\n===== {name} =====")
        model = build_model(name, args.pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

        best_score = -1.0
        best_path = models_dir / f"{name}.pth"
        history = []
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
            valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, device, False)
            scheduler.step(valid_metrics["f1"])
            row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
            history.append(row)
            print(
                f"epoch {epoch}/{args.epochs} | "
                f"train f1={train_metrics['f1']:.4f} | valid f1={valid_metrics['f1']:.4f} | "
                f"valid auc={valid_metrics['roc_auc']:.4f}"
            )
            if valid_metrics["f1"] > best_score:
                best_score = valid_metrics["f1"]
                torch.save({
                    "model_name": name,
                    "class_to_idx": {"DR": 0, "No_DR": 1},
                    "state_dict": model.state_dict(),
                }, best_path)

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, False)
        elapsed = round(time.time() - start, 2)
        summary["models"][name] = {
            "weights": str(best_path),
            "best_valid_f1": best_score,
            "test": test_metrics,
            "seconds": elapsed,
            "history": history,
        }
        print(f"TEST: {test_metrics}")

    with open(models_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"\nDone. Weights and summary saved in: {models_dir}")


if __name__ == "__main__":
    main()
