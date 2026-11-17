# Real Time Driver Drowsiness Detection System and Alert System

An AI-powered Driver Drowsiness Detection System built using **Deep Learning**, **TensorFlow**, **OpenCV**, and **LSTM networks** to detect driver fatigue in real time.

The project analyzes video sequences of drivers and predicts whether the driver is **drowsy** or **alert**, helping improve road safety and reduce accident risks. After detection if the dirver is found to be drowsy then a sound alert is generated to wake them up.

---

# Features

* Real-time video frame processing
* Deep Learning-based drowsiness classification
* Sequence learning using LSTM
* Frame extraction and preprocessing using OpenCV
* Accuracy and loss visualization
* Confusion matrix and evaluation metrics
* Model checkpoint saving and loading
* Audio alert system for drowsiness detection

---

# Tech Stack

## Languages & Frameworks

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# Deep Learning Architecture

The model uses:

* **TimeDistributed Conv2D Layers** for spatial feature extraction
* **Batch Normalization** for stable training
* **MaxPooling Layers** for dimensionality reduction
* **LSTM Layers** for temporal sequence learning
* **Dense Layers** for final classification

---

# Dataset

Dataset used:

SUST Driver Drowsiness Dataset

Downloaded using:

```python
import kagglehub
path = kagglehub.dataset_download("esrakavalci/sust-ddd")
```

Dataset contains:

* Drowsiness videos
* Non-drowsiness videos

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd Driver_Drowsiness_Detection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pillow kagglehub scipy
```


---

# Preprocessing Pipeline

1. Video input is loaded
2. Frames are extracted using OpenCV
3. Frames are resized to:

```python
IMG_SIZE = (64, 64)
```

4. Sequences are generated:

```python
SEQUENCE_LENGTH = 30
```

5. Data is normalized and passed to the model

---

# Model Training

The project uses:

* Early Stopping
* Model Checkpointing
* Validation Split
* Accuracy & Loss Monitoring

---

# Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# Real-Time Detection

The system supports:

* Webcam-based monitoring
* Live prediction
* Audio warning alerts when drowsiness is detected

---

# Sample Workflow

```text
Video Input → Frame Extraction → CNN Feature Extraction → LSTM Sequence Learning → Drowsy / Alert Prediction
```

---



# Learning Outcomes

This project demonstrates:

* Computer Vision
* Deep Learning
* Sequence Modeling
* Video Processing
* Real-time AI Systems

---

# Author

## Manasvi Jain


 If You Like This Project

Give this repository a ⭐ on GitHub and support the project.


