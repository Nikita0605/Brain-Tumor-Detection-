# MRI Brain Tumor Detection

This project focuses on detecting brain tumors using MRI images. The dataset used for training and testing the model can be found on Kaggle:

1. [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
2. [Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

## Table of Contents

- [Installation](#installation)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Dataset Class](#dataset-class)
- [Data Augmentation and Visualization](#data-augmentation-and-visualization)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Evaluation](#evaluation)
- [Feature Map Visualization](#feature-map-visualization)
- [Handling Overfitting](#handling-overfitting)
- [Conclusion](#conclusion)

## Installation

To run this project, you'll need to install the required packages. This can be done using pip:

```bash
!pip install torch monai keras
```

## Data Loading and Preprocessing

The MRI images are loaded from the specified directories and resized to 128x128 pixels. The images are then split into tumor and healthy categories, normalized, and prepared for model training.

## Dataset Class

A custom dataset class `MRI` is created to handle the loading and preprocessing of MRI images. This class supports both training and validation modes, enabling easy splitting of the dataset.

## Data Augmentation and Visualization

The project includes functions for data augmentation and visualization. A function `plot_random` is used to plot random images from the healthy and tumor categories to visualize the data.

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) implemented using PyTorch. The architecture includes convolutional layers, average pooling layers, dropout layers for regularization, and fully connected layers for classification.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.BatchNorm2d(16)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1),
            nn.Sigmoid()
        )
```

## Training and Validation

The model is trained using the Adam optimizer and binary cross-entropy loss function. The training process includes both training and validation phases to monitor and prevent overfitting. L2 regularization is also applied during training.

## Evaluation

The trained model is evaluated on a test set, and the performance metrics such as accuracy, precision, recall, and specificity are calculated.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

test_accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
```

## Feature Map Visualization

The feature maps of the convolutional layers are visualized to understand the internal workings of the CNN. This helps in interpreting how the model processes and identifies features in the MRI images.

## Handling Overfitting

To handle overfitting, the project includes techniques like dropout, data augmentation, and early stopping. The training and validation loss curves are plotted to monitor the model's performance over epochs.

## Conclusion

This project demonstrates the use of CNNs for brain tumor detection from MRI images. The implementation includes data loading, preprocessing, model architecture design, training, evaluation, and visualization. The results show the effectiveness of CNNs in medical image classification tasks.

---

