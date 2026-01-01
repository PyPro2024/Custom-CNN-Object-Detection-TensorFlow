# Custom Object Detection Model with TensorFlow

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-green)

##  Project Overview
This project implements a **Custom Object Detection Architecture** from the ground up using **TensorFlow** and **Keras**. Instead of relying solely on out-of-the-box API calls, this project constructs the neural network backbone layer-by-layer to understand the fundamentals of feature extraction and bounding box regression.

The project is divided into three key phases:
1.  **Backbone Construction:** Designing a custom Convolutional Neural Network (CNN) to serve as the feature extractor.
2.  **Head Design:** Implementing the Classification Head (what is the object?) and Regression Head (where is the object?).
3.  **Transfer Learning:** Utilizing **TensorFlow Hub** to integrate pre-trained models and standard datasets (COCO/Pascal VOC) for robust performance.

##  Model Architecture
The custom model follows a classic single-stage detector design:

* **Input Layer:** Accepts RGB images (resized to 224x224).
* **Feature Extractor (Backbone):**
    * Stack of `Conv2D` layers (32, 64 filters) with ReLU activation.
    * `MaxPooling2D` for spatial dimensionality reduction.
    * Designed to capture hierarchical patterns (edges -> shapes -> objects).
* **Detection Heads:**
    * **Classification:** Dense layer with Softmax activation to predict class probabilities (e.g., Person, Car, Dog).
    * **Regression:** Dense layer with linear/sigmoid activation to predict Bounding Box coordinates $(x, y, w, h)$.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras, TensorFlow Hub
* **Image Processing:** OpenCV (`cv2`), PIL, NumPy
* **Data Handling:** XML Parsing (`xml.etree`) for dataset annotations.
* **Visualization:** Matplotlib

##  Data Pipeline
The project includes a robust data pipeline capable of:
* Mounting external storage (Google Drive).
* Parsing **XML Annotations** (common in Pascal VOC datasets).
* Mapping class labels to the standard **COCO dataset** schema.

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PyPro2024/Custom-CNN-Object-Detection-TensorFlow.git]
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy opencv-python matplotlib tensorflow-hub
    ```
3.  **Run the Notebook:**
    Open `Object_Det_CustomModel.ipynb` in Jupyter Notebook or Google Colab.
    * *Note: Ensure you update the dataset paths if running locally.*

##  Results
* **Parameter Efficiency:** Investigated the impact of adding convolutional layers on total parameter count and model complexity.
* **Inference:** demonstrated successful forward passes predicting bounding box coordinates and class labels.

---
*If you find this project helpful for understanding Computer Vision internals, feel free to ⭐ the repo!*
