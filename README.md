# Chest X-Ray Pneumonia Detection

## Overview

This project develops and evaluates deep learning models for automated pneumonia detection from chest X-ray images. A custom Convolutional Neural Network (CNN) is compared against two transfer learning architectures, EfficientNetB0 and ResNet50, using a publicly available medical imaging dataset.

Beyond standard classification performance, the project investigates model interpretability through Grad-CAM visualizations, threshold-based decision analysis, and detailed error analysis to better understand clinically relevant prediction behavior.

---

## Project Highlights

* Custom CNN baseline architecture
* Transfer Learning with EfficientNetB0 and ResNet50
* Stratified train-validation splitting
* Conservative medical-image augmentation
* Class imbalance handling through class weighting
* ROC-AUC and Precision-Recall evaluation
* Threshold analysis for FP/FN trade-offs
* Grad-CAM explainability
* Misclassification and error analysis
* Model comparison and reporting

---

## Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

### Original Dataset Distribution

| Split      |    Normal | Pneumonia |     Total |
| ---------- | --------: | --------: | --------: |
| Train      |     1,341 |     3,875 |     5,216 |
| Validation |         8 |         8 |        16 |
| Test       |       234 |       390 |       624 |
| **Total**  | **1,583** | **4,273** | **5,856** |

The original validation set contains only 16 images, making it unsuitable for reliable model selection, early stopping, and learning-rate scheduling.

Therefore, a **stratified 80/20 split** of the original training set is used to create a statistically meaningful validation subset while keeping the original test set untouched for final evaluation.

Resulting dataset sizes:

| Subset     | Images |
| ---------- | -----: |
| Train      |  4,173 |
| Validation |  1,043 |
| Test       |    624 |

---

## Research Objectives

This project investigates the following questions:

1. How effectively can a custom CNN classify pneumonia from chest X-ray images?
2. Do transfer learning models outperform a custom CNN?
3. Can class weighting reduce the impact of class imbalance?
4. How does threshold selection affect false positive and false negative rates?
5. What insights can Grad-CAM provide regarding model decision-making?
6. Does conservative data augmentation improve generalization?
7. What patterns are present in misclassified chest X-rays?

---

## Workflow

The complete project workflow is illustrated below.

<p align="center">
  <img src="workflow_pneumonia_detection.png" width="100%">
</p>

The workflow includes:

1. Dataset preparation
2. Data exploration
3. Image preprocessing
4. Conservative medical augmentation
5. Model development
6. Model training
7. Evaluation
8. Threshold analysis
9. Grad-CAM explainability
10. Error analysis
11. Final model comparison and reporting

---

## Data Preprocessing

All chest X-ray images are:

* Converted to RGB format
* Resized to 224 × 224 pixels
* Split using stratified sampling

Model-specific preprocessing is applied:

| Model          | Preprocessing                 |
| -------------- | ----------------------------- |
| Custom CNN     | Pixel normalization [0,1]     |
| EfficientNetB0 | EfficientNet preprocess_input |
| ResNet50       | ResNet preprocess_input       |

---

## Data Augmentation

A conservative augmentation strategy is applied only to the training set to preserve clinically meaningful radiographic structures.

### Applied Augmentations

* Rotation (±8°)
* Width Shift (±5%)
* Height Shift (±5%)
* Zoom (±8%)

### Excluded Augmentations

* Brightness augmentation
* Horizontal flipping
* Shear transformations

These operations were intentionally avoided because they may alter diagnostically important features in chest radiographs.

---

## Models

### 1. Custom CNN

A lightweight CNN designed specifically for pneumonia classification.

Architecture components:

* Convolutional Blocks
* Batch Normalization
* Max Pooling
* Global Average Pooling
* Dense Layers
* Sigmoid Output

### 2. EfficientNetB0

ImageNet-pretrained EfficientNetB0 using transfer learning.

Training strategy:

* Feature extraction
* Controlled fine-tuning
* Reduced learning-rate scheduling

### 3. ResNet50

ImageNet-pretrained ResNet50 used as a deeper transfer-learning baseline.

Training strategy:

* Frozen backbone training
* Conservative fine-tuning
* Additional regularization to reduce overfitting

---

## Training Strategy

The following techniques are used across models:

* Binary Cross-Entropy Loss
* Adam Optimizer
* Class Weighting
* Early Stopping
* ReduceLROnPlateau
* Model Checkpointing

Transfer learning models are trained in two stages:

### Stage 1 – Feature Extraction

Pretrained backbone remains frozen while the classification head is trained.

### Stage 2 – Fine-Tuning

Selected upper layers are unfrozen and optimized using a smaller learning rate.

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1-Score
* ROC-AUC
* Average Precision (AP)
* Confusion Matrix
* Precision-Recall Curve

Particular attention is given to:

* False Negative reduction
* Recall performance
* Clinical reliability

Since missing pneumonia cases is generally more critical than generating additional false positives.

---

## Threshold Analysis

In addition to conventional evaluation, probability-threshold analysis is performed to investigate the trade-off between:

* False Positives (FP)
* False Negatives (FN)
* Precision
* Recall

This provides additional insight into how decision thresholds influence model behavior in a medical context.

---

## Explainability (Grad-CAM)

Grad-CAM visualizations are generated for:

* Custom CNN
* EfficientNetB0
* ResNet50

The explainability analysis is used to:

* Identify influential image regions
* Assess model attention patterns
* Verify clinically meaningful focus areas
* Compare interpretability across architectures

---

## Error Analysis

Misclassified cases are investigated through:

### False Positives

Normal images incorrectly predicted as pneumonia.

### False Negatives

Pneumonia images incorrectly predicted as normal.

### Misclassification Review

Analysis of challenging chest X-rays and recurring failure patterns.

This step provides insights beyond aggregate performance metrics.

---

## Repository Structure

```text
.
├── chest_xray_pneumonia_detection_final_optimized.ipynb
├── README.md
├── workflow_pneumonia_detection.png
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── tables/
│   └── reports/
│
└── dataset/
```

---

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* OpenCV

---

## Author

**Elif Sila Okutucu**

Machine Learning • Deep Learning • Computer Vision

University of Europe for Applied Sciences
