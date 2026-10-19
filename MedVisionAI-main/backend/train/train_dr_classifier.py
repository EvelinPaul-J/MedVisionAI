"""
train_dr_classifier.py  (SCAFFOLD — requires torch, torchvision, and a
labeled dataset such as IDRiD / APTOS 2019 laid out as:

    dataset/
      train/0_no_dr/*.jpg
      train/1_mild/*.jpg
      train/2_moderate/*.jpg
      train/3_severe/*.jpg
      train/4_pdr/*.jpg
      val/...  (same structure)

Not run automatically by the app — see train/README.md.
Install first:  pip install torch torchvision
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = "dataset"
NUM_CLASSES = 5
EPOCHS = 15
BATCH_SIZE = 16
LR = 3e-4


def build_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss/len(train_ds):.4f} - val_acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "dr_efficientnet_b3_best.pt")

    print("Training complete. Best val acc:", best_acc)


if __name__ == "__main__":
    main()
