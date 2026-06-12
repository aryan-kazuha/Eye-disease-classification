# Eye Disease Classification using CNN and VGG16

An end-to-end Deep Learning and Computer Vision pipeline designed to automate the multi-class classification of retinal diseases from optical fundus imagery. The project implements a custom Convolutional Neural Network (CNN) and leverages Transfer Learning using a fine-tuned VGG16 architecture to significantly boost diagnostic accuracy while mitigating overfitting.

## Core Features

* **Dual-Model Architecture:** Implements a baseline custom CNN alongside a fine-tuned VGG16 backbone initialized with ImageNet weights.
* **Transfer Learning Optimization:** Replaces top-level dense layers with custom classifier heads to adapt a pre-trained feature extractor to clinical optical imagery.
* **Robust Image Augmentation:** Integrates real-time spatial transformations (rotation, zoom, shear) within the preprocessing pipeline to enhance model generalization.
* **Advanced Regularization:** Incorporates Batch Normalization, Dropout layers, and dynamic learning rate callbacks to stabilize gradient descent and combat training stagnation.

---

## Technical Stack

* **Deep Learning Framework:** TensorFlow, Keras
* **Computer Vision & Processing:** OpenCV (`cv2`), NumPy, Pandas
* **Data Ingestion:** `opendatasets` (Kaggle API integration)
* **Visualization:** Matplotlib, Seaborn

---

## Dataset Pipeline

The pipeline automatically fetches the **Eye Disease Image Dataset** directly from Kaggle, which includes balanced clinical optical fundus photographs across multiple specialized categories (such as Central Serous Chorioretinopathy and other retinal anomalies).

* **Resolution Scaling:** Images are parsed and systematically resized using OpenCV to match the native `(224, 224, 3)` dimensional constraints of the VGG16 input tensor.
* **Data Augmentation:** Real-time data pipeline configuration applying random scaling, zooming, and shearing matrices to increase dataset variance.

---

## Architecture Overview

### 1. Custom CNN Baseline
A multi-layered sequence featuring structural blocks of:
`Conv2D` -> `BatchNormalization` -> `MaxPooling2D` -> `Dropout` -> `Flatten` -> `Dense`

### 2. VGG16 Transfer Learning Pipeline
* **Base Network:** Frozen convolutional base of VGG16 (`include_top=False`) retaining deep feature extractors learned from the ImageNet dataset.
* **Classifier Head:** Custom Dense layers integrated with Dropout nodes to classify targeted ocular pathologies.

---

## Model Performance Metrics

Leveraging pre-trained deep visual features drastically reduced training convergence times and enhanced performance across validation and test thresholds:

| Metric | Custom Baseline CNN | Fine-Tuned VGG16 Network |
| :--- | :--- | :--- |
| **Initial / Baseline Accuracy** | ~46% | — |
| **Final Classification Accuracy** | — | **~74%** |

---
