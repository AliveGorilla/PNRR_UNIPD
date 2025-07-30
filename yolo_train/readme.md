# 🚀 YOLO Training Script (Ultralytics)

This repository contains a script for training YOLO models using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) implementation. The script automates model training, evaluation, and exporting results, and is intended for use with labeled image datasets (e.g., architectural elements like windows and balconies).

---

## 🧠 Features

- Automatic training using a given `data.yaml` file
- Custom augmentations (shear, flip)
- Best checkpoint is copied to `final.pt` automatically
- Evaluates and prints metrics: mAP, Precision, Recall
- All outputs saved to an organized `output/` folder

---

## 📁 Dataset Structure

The dataset folder must follow this structure:

```
input_folder/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── classes.txt
├── data.yaml
├── notes.json   # optional metadata
```

### 📄 `data.yaml` Example

```yaml
path: data_path
train: images/train
val: images/val

names:
  0: class_0
  1: class_1
  2: class_2
```

---

## 🧪 How to Run

### 📦 Install Dependencies

Make sure you have Python 3.8+ and the Ultralytics package installed:

```bash
pip install ultralytics
```

### ▶️ Run Training

```bash
python train_yolo.py --dataset_path path/to/your/input_folder
```

- `--dataset_path`: Path to the folder containing `data.yaml`

The script will:
- Train a model from the specified `data.yaml`
- Save output in `output/yolo_11_windows/`
- Evaluate the best checkpoint
- Print results like:

```
>>> Final model saved to: output/yolo_11_windows/final.pt

? Evaluation results:
  mAP50:     0.9900
  mAP50-95:  0.7800
  Precision: 0.9700
  Recall:    0.9800
```

---

## ⚙️ Model Config

Inside the script, you can adjust the following training parameters:

```python
model = YOLO('yolo11l.pt')  # Load pretrained model
model.train(
    seed=4,
    epochs=100,
    imgsz=640,
    batch=32,
    augment=True,
    shear=10,
    flipud=0.5,
    fliplr=0.7,
    ...
)
```

Replace `'yolo11l.pt'` with any custom checkpoint or YOLO variant like `yolov11n.pt`, `yolov11m.pt`, etc.

---

## 📤 Output

- Model checkpoints: `output/yolo_11_windows/weights/`
- Final model: `final.pt`
- Validation metrics printed to console
- Logs saved in project directory

---

### 🖼️ Graphs

<div style="display: flex; justify-content: space-between; gap: 20px;">
  <div style="text-align: center;">
    <img src="/yolo_train/plans_detection_results/val_batch1_labels.jpg" alt="Plans Detection" width="400"/>
    <p><strong>Plans Detection</strong></p>
  </div>
  <div style="text-align: center;">
    <img src="/yolo_train/windows_detection_results/val_batch1_labels.jpg" alt="Windows Detection" width="300"/>
    <p><strong>Windows Detection</strong></p>
  </div>
</div>

---

## 📌 Notes

- Uses the official [Ultralytics API](https://docs.ultralytics.com) for training and evaluation.
- Assumes YOLO-style annotations (one `.txt` file per image in `labels/`).

---
