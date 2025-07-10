# Alzheimer-s-Disease-Classification
This project classifies MRI brain scans into 4 Alzheimer’s stages using deep learning models, including CNN, DCNN, and an enhanced Wavelet Convolutional Neural Network (WCNN). It uses Grad-CAM for explainability and SMOTE for data balancing.

📚 Dataset
Source: Kaggle Dataset
Total Images: 6,400 MRI scans
Classes:
0: Mild Demented
1: Moderate Demented
2: Non Demented
3: Very Mild Demented
Format: .parquet files, loaded and decoded from byte format
Resized To: 128×128 grayscale images
🧪 Preprocessing Steps
Byte decoding & resizing
Normalization to [0, 1]
Stratified train-validation-test split
Class balancing using SMOTE
One-hot encoding of labels
Final shape: (128, 128, 1) for CNN/DCNN, (64, 64, 1) for WCNN after DWT
🧠 Models Implemented
Model	Accuracy (Test)	Notes
CNN	84.53%	With SE blocks and Avg-TopK Pooling
DCNN	93.44%	Deeper version with SE + Strided Conv
WCNN	95.63%	Best model using Haar DWT
⚙️ WCNN Highlights
Uses Discrete Wavelet Transform (DWT) for downsampling
Retains only LL (approximation) coefficients
Combines wavelet + CNN power for better spatial-frequency extraction
Simple yet accurate – fewer layers, no SE blocks
🛰️ Explainability
Used Grad-CAM to visualize class activation regions in MRI scans
Highlights relevant brain regions per Alzheimer’s stage
Interpretable and clinically meaningful outputs
🔧 Technologies Used
Python 3.7+
TensorFlow / Keras
Pandas, NumPy
Matplotlib, Seaborn
OpenCV
imbalanced-learn (for SMOTE)
tf-keras-vis (for Grad-CAM)
