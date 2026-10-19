"""
train_lesion_detector.py  (SCAFFOLD — requires ultralytics + a
YOLO-format lesion dataset with classes: microaneurysm, hemorrhage,
hard_exudate, cotton_wool_spot)

Not run automatically by the app — see train/README.md.
Install first:  pip install ultralytics

Expected dataset.yaml:
    path: dataset
    train: images/train
    val: images/val
    names:
      0: microaneurysm
      1: hemorrhage
      2: hard_exudate
      3: cotton_wool_spot
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # start from a small pretrained checkpoint
    model.train(
        data="dataset.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        name="retina_lesion_yolo",
    )
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
