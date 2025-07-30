# 🏠 House Facade Direction Dataset Generator

This project aims to automatically generate a structured dataset of houses, their windows and balconies, and label them with their corresponding **cardinal direction** (e.g., North, East) based on the compass present in architectural schema files (usually in PDF format).

The pipeline leverages a combination of object detection (YOLO) and deep learning (ResNet18) to extract relevant visual information and assign directional labels.

---

## 📁 Project Overview

The dataset generation pipeline is divided into **three main stages**:

### 1. Detecting Schema and Compass from PDF

- Input: PDF architectural files (converted to JPG during preprocessing).
- A pretrained **YOLO model** is used to detect:
  - The **building schema** area.
  - The **compass** area, indicating directions (usually a North arrow).
- Outputs: Cropped schema and compass images.

### 2. Detecting Windows and Balconies

- Another **YOLO model** is applied to the **cropped schema** to detect:
  - **Windows**
  - **Balconies**
- The bounding boxes of these elements are extracted for directional labeling.

### 3. Estimating Compass Orientation with ResNet18

- A **ResNet18** classification model is trained to predict the **North direction** on the cropped compass image.
- The predicted direction is projected onto the building schema.
- Each detected **window** and **balcony** is assigned a direction label based on its position relative to the estimated North.

---

## 📦 Output

- A labeled dataset of building elements (windows, balconies) with direction tags (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`).
- Optionally, annotated schema images for visual verification.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- YOLOv5 / YOLOv8 (for object detection)
- ResNet18 (for direction classification)
- OpenCV, NumPy, Matplotlib
- PDF to Image Conversion tools

---

## 🧠 Potential Applications

- Urban modeling and simulation
- Environmental analysis (sunlight exposure, ventilation)
- Smart city planning
- Real estate insights

---

## 🚧 Future Work

- Improve compass orientation classification accuracy.
- Integrate support for multiple compass types and symbols.
- Extend YOLO detection to doors or other facade elements.
- Export final dataset in popular formats (e.g., COCO, CSV, GeoJSON).

---

## 🤝 Contributions

Pull requests are welcome. If you have suggestions or improvements, feel free to open an issue or fork the repository.
