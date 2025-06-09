

# Knee X-Ray Classification Using CNN & Transfer Learning

---

## 🚀 Project Overview

This project implements an end-to-end pipeline to classify **knee X-ray images** into three categories:

* **Normal**
* **Osteopenia**
* **Osteoporosis**

using deep learning techniques, including a custom Convolutional Neural Network (CNN) and a Transfer Learning approach with **ResNet50**.

---

## 📂 Dataset

* Dataset consists of knee X-ray images organized in nested ZIP archives.
* Classes: `normal`, `osteopenia`, `osteoporosis`.
* Automatically downloads, extracts, and prepares the data.
* Splits dataset into training, validation, and testing sets with stratification.

---

## ⚙️ Features

* **Data Augmentation** for robust model generalization:
  Rotation, shifting, zooming, flipping, brightness adjustments.
* **Two model architectures**:

  * Custom CNN built from scratch
  * Transfer Learning using pretrained ResNet50
* **Comprehensive evaluation**:
  Accuracy, precision, recall, F1-score, confusion matrix.
* **Training visualization**:
  Plots of accuracy and loss over epochs.
* **Automatic cleanup**:
  Removes temporary files and directories after execution.

---

## 🛠 Installation & Setup

Run the following command to install dependencies:

```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn tqdm
```

---

## 🔧 Usage

1. Run the notebook or Python script in an environment with internet access.
2. The code automatically:

   * Downloads and extracts the dataset.
   * Prepares train/validation/test splits.
   * Trains the models with augmentation.
   * Evaluates and visualizes results.
3. Results and plots are displayed inline.

---

## 🧠 Model Details

### Custom CNN

* Multiple Conv2D layers with Batch Normalization & Dropout
* MaxPooling layers to reduce spatial dimensions
* Dense layers with ReLU activation
* Output layer with Softmax for multi-class classification

### Transfer Learning with ResNet50

* Pretrained ResNet50 backbone (imagenet weights)
* Frozen convolutional base for feature extraction
* Custom classifier head with Dense and Dropout layers

---

## 📊 Evaluation Metrics

| Metric    | Description                                              |
| --------- | -------------------------------------------------------- |
| Accuracy  | Overall correct classification rate                      |
| Precision | Correct positive predictions / total predicted positives |
| Recall    | Correct positive predictions / total actual positives    |
| F1-Score  | Harmonic mean of Precision and Recall                    |

Includes a detailed classification report and confusion matrix.

---

## 📈 Visualization

* Training vs validation accuracy over epochs
* Training vs validation loss over epochs
* Confusion matrix heatmap

---

## 🧹 Cleanup

* Temporary files and extracted folders are deleted automatically post-training to save disk space.

---

## 📌 Requirements

* Python 3.7+
* TensorFlow 2.x
* scikit-learn
* pandas
* matplotlib
* seaborn
* tqdm






