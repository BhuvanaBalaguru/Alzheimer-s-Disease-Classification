# Alzheimer’s Disease Classification 🧠

This project classifies MRI brain scans into **four stages of Alzheimer’s disease** using deep learning models:  
- 🧩 Convolutional Neural Network (CNN)  
- 🧱 Deep CNN (DCNN)  
- 🌊 Wavelet CNN (WCNN)

It also uses **Grad-CAM for explainability** and **SMOTE for class balancing**.

---

## 📚 Dataset

- **Source**: Kaggle  
- **Total Images**: 6,400 MRI scans  
- **Classes**:
  - `0`: Mild Demented  
  - `1`: Moderate Demented  
  - `2`: Non Demented  
  - `3`: Very Mild Demented  
- **Format**: `.parquet` files (loaded and decoded from byte format)  
- **Image Size**: Resized to `128×128` grayscale

---

## 🧪 Preprocessing Steps

- Byte decoding from `.parquet`
- Resizing to `128x128` for CNN/DCNN, `64x64` for WCNN after wavelet transform
- Normalization to range `[0, 1]`
- Stratified train-validation-test split
- SMOTE applied for class balancing
- One-hot encoding of labels

---

## 🧠 Models Implemented

| Model | Accuracy (Test) | Highlights |
|-------|-----------------|------------|
| **CNN** | 84.53% | SE Blocks + Avg-TopK Pooling |
| **DCNN** | 93.44% | Deeper with SE + Strided Conv |
| **WCNN** | **95.63%** | Haar DWT + Lightweight + Most Accurate |

---

## ⚙️ WCNN Highlights

- Uses **Discrete Wavelet Transform (DWT)** for downsampling
- Retains only LL (approximation) coefficients
- Captures both **spatial and frequency** features
- Fewer layers than DCNN (no SE blocks), but **more accurate**

---

## 🛰️ Explainability: Grad-CAM

- Applied **Grad-CAM** to visualize key regions of attention in MRI scans
- Helps interpret which parts of the brain the model focuses on
- Clinically meaningful visualizations

---

## 🔧 Technologies Used

- Python 3.7+
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- OpenCV
- `imbalanced-learn` (SMOTE)
- `tf-keras-vis` (Grad-CAM)

---

