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
  mAP50
