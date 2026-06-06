# Chest X-Ray Pneumonia Detection

**Course:** PS26 — DSC01 Machine Learning  
**Author:** Elif Sila Okutucu  
**Institution:** University of Europe for Applied Sciences  

---

## Project Overview

This project develops and evaluates deep learning models for automated pneumonia detection from chest X-ray images. A custom CNN architecture is compared against two transfer learning models (EfficientNetB0 and ResNet50) using a publicly available Kaggle dataset.

The project addresses class imbalance, applies data augmentation, implements Grad-CAM explainability, and performs error analysis across all three models.

---

## Dataset

**Name:** Chest X-Ray Images (Pneumonia)  
**Source:** [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  

| Split | Normal | Pneumonia | Total |
|-------|-------:|----------:|------:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

> The original validation set (16 images) is too small for reliable evaluation. A stratified 80/20 split of the training set is used instead.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | How effectively can a custom CNN classify pneumonia from chest X-ray images? |
| RQ2 | To what extent do transfer learning models outperform a custom CNN across accuracy, recall, F1-score, and AUC-ROC? |
| RQ3 | Can class weighting mitigate the impact of class imbalance on false negative predictions? |
| RQ4 | How do different evaluation metrics influence the assessment of pneumonia classification performance? |
| RQ5 | What insights do Grad-CAM visualizations provide about the regions influencing model predictions? |
| RQ6 | To what extent does data augmentation improve generalization and reduce overfitting? |
| RQ7 | What characteristics distinguish misclassified chest X-rays from correctly classified cases? |

---

## Repository Structure
