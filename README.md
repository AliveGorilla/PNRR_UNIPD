## 🏠 House Facade Direction Dataset Generator

This project aims to automatically generate a structured dataset of houses, their windows and balconies, and label them with their corresponding **cardinal direction** (e.g., North, East) based on the compass present in architectural schema files (usually in PDF format).

The pipeline leverages a combination of object detection (YOLO) and deep learning (ResNet18) to extract relevant visual information and assign directional labels.

---

## 📁 Project Overview

The dataset generation pipeline is divided into **three main stages**:

### 1. Detecting Schema and Compass from PDF

- Input: PDF architectural files (converted to JPG during preprocessing).
- Train [YOLO](https://github.com/ultralytics/ultralytics) for object detection:
  - The **inner schema** area.
  - The **outside schema** area (major samples of the data have several floors in one file and not living areas are not in the scope of interes).
  - The **compass** area, indicating directions (usually a North arrow).
- Outputs: Cropped schema and compass images.

### 2. Detecting Windows and Balconies

- Another [YOLO](https://github.com/ultralytics/ultralytics) is applied to the **cropped schema** to detect:
  - **Windows in** (windows inside of the living area).
  - **Windows out** (the rest).
  - **Balconies**.
- The bounding boxes of these elements are extracted for directional labeling.

### 3. Estimating Compass Orientation with ResNet18

- A **ResNet18** classification model is trained to predict the **North direction** and **Compass center** on the cropped compass image.
- Build 8 axis (North, North-West, South-West, etc.)
- The predicted direction is projected onto the building schema.
- Each detected **Window in** and **Balcony** is assigned a direction label based on its position relative to the estimated North.

---

## ✍️ Data Annotation

- **Manual labeling** was performed using [Label Studio](https://labelstud.io/).
- Annotations include bounding boxes for schemas, compasses, windows, and balconies.
- Labels were used to train both YOLO object detection models and ResNet direction classifier.

---

## 📊 Sample Dataset Structure

Below is an example of how the generated dataset is structured:

| File Name        | Window Directions    | # Windows | Balcony Directions | # Balconies | Error Code | Main Direction |
|------------------|----------------------|-----------|---------------------|--------------|-------------|----------------|
| house_001.jpg    | N, NE, E             | 3         | NE                  | 1            | 0           | N              |
| house_002.jpg    | S, SW                | 2         | SW, W               | 2            | 1           | SW             |
| house_003.jpg    | E, SE, S             | 3         | -                   | 0            | 0           | SE             |

### ⚠️ Error Code Legend

- **0** — No problem  
- **1** — No plan was found AND no windows were found  
- **2** — Plan was found BUT no compass was found  
- **3** — Both compass and plan were found BUT no windows

---

## 📈 Results

The models achieved strong performance across all stages of the pipeline:

### 🧭 [YOLO Models — Schema, Compass, Window & Balcony Detection](https://github.com/AliveGorilla/PNRR_UNIPD/tree/main/yolo_train)

| Metric             | Value |
|--------------------|-------|
| Precision          | 0.97  |
| Recall             | 0.98  |
| mAP50              | 0.99  |
| mAP50-95           | 0.78  |

*Evaluation performed on a held-out validation set using the Ultralytics YOLO implementation.*

### 🧠 [ResNet18 — Direction (North, Center) Classification](https://github.com/AliveGorilla/PNRR_UNIPD/tree/main/compass_direction_train)

- **Direction prediction accuracy**: ~98%

This model classifies the North direction on cropped compass images, enabling reliable orientation mapping for windows and balconies.

---

## 🛠️ Tech Stack

- Python
- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (for object detection)
- ResNet18 via PyTorch (for direction classification)
- OpenCV, NumPy, Matplotlib
- [Label Studio](https://labelstud.io/) (for manual annotation)
- PDF to Image Conversion tools

---

## 🧠 Potential Applications

- Urban modeling and simulation
- Environmental analysis (sunlight exposure, ventilation)
- Smart city planning
- Real estate insights

---

## 🚧 Future Work

- Improve windwos/balcony detection accuracy.
- Integrate support for multiple compass types and symbols.
- Extend YOLO detection to doors or other facade elements.
- Export final dataset in popular formats (e.g., COCO, CSV, GeoJSON).

---

### 🖼️ Example Visualization
![](/example.png)

---

## 🤝 Contributions

Pull requests are welcome. If you have suggestions or improvements, feel free to open an issue or fork the repository.


