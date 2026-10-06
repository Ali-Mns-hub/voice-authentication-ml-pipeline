# 🎙️ Voice Authentication & Gender Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised%20%26%20Unsupervised-orange)
![Audio Processing](https://img.shields.io/badge/Audio%20Processing-Librosa-green)
![Institution](https://img.shields.io/badge/Institution-University%20of%20Tehran-red)

An end-to-end Machine Learning pipeline for processing raw audio signals, extracting complex acoustic features, and training models for **Gender Classification** and **Closed-Set Speaker Authentication**. This repository also explores hidden acoustic patterns using dimensionality reduction (PCA, t-SNE) and unsupervised clustering (K-Means++, GMM).

> **Academic Context:** Developed as the Final Project for the Machine Learning Course at the Department of Electrical and Computer Engineering, University of Tehran (Fall 2024)[cite: 121].

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Data Pipeline & Signal Processing](#-data-pipeline--signal-processing)
3. [Feature Extraction](#-feature-extraction)
4. [Supervised Learning (Classification)](#-supervised-learning-classification)
5. [Unsupervised Learning (Clustering)](#-unsupervised-learning-clustering)
6. [Key Results](#-key-results)
7. [Installation & Usage](#-installation--usage)
8. [Repository Structure](#-repository-structure)

---

## 📖 Project Overview
This project processes non-uniform raw `.mp3` audio recordings to solve two primary tasks:
*   **Gender Classification:** A binary classification task to identify the speaker's gender based on extracted acoustic patterns[cite: 124, 131].
*   **Closed-Set Speaker Authentication:** A multi-class classification system that authenticates registered speakers. To handle real-world class imbalances, a rigorous **Balanced Sampling** strategy was implemented[cite: 131, 144, 207].

---

## ⚙️ Data Pipeline & Signal Processing
Working directly with raw audio requires robust digital signal processing (DSP) before any machine learning can occur. The `AudioPreprocessor` module handles[cite: 175, 176]:

*   **Noise Reduction:** Implemented both **Spectral Subtraction** (using STFT and neural filtering) and **Bandpass Filtering** (100Hz - 8000Hz) to remove background noise[cite: 176].
*   **Resampling & Windowing:** Standardized all audio to a `22050 Hz` sample rate. Signals were chunked into discrete overlapping frames (Window Size: 2048, Hop Length: 512) for granular analysis[cite: 175, 176].
*   **Amplitude Normalization:** Applied volume normalization to ensure uniformity across different recording environments[cite: 176].

*Comparison of Mel-Spectrogram before and after Spectral Subtraction Denoising:*
<br>
![Denoising Visual](assets/spectrogram_denoised.png)

---

## 🎶 Feature Extraction
Using the `librosa` library, the `AudioFeatureExtractor` transforms raw audio frames into dense numerical representations. The extracted features (saved to `features.csv`) include[cite: 185, 186]:

1.  **MFCCs (Mel Frequency Cepstral Coefficients):** Captures the human auditory system's response.
2.  **Mel-Spectrogram & Log Spectrogram:** Frequency domain power analysis[cite: 186].
3.  **Spectral Features:** Spectral Centroid (center of mass), Bandwidth, and Contrast (peaks vs. valleys)[cite: 186, 187].
4.  **Temporal Features:** Zero-Crossing Rate (ZCR) and Total Energy[cite: 187].

---

## 🧠 Supervised Learning (Classification)
Five distinct models were trained and evaluated: **Logistic Regression, SVM (Linear Kernel), KNN (k=3), MLP (100 hidden neurons), and XGBoost**[cite: 195, 197, 198, 199]. All features were standardized using `StandardScaler` prior to training.

### Task 1: Gender Classification
*   **Accuracy Champion:** The **MLP (Multi-Layer Perceptron)** achieved the highest overall accuracy at **93.8%**, excelling in learning non-linear relationships[cite: 202, 206].
*   **Separability Champion:** Analysis of the ROC curve revealed that **XGBoost** achieved the highest Area Under Curve (**AUC = 0.68**), making it the most robust model for distinguishing between class boundaries[cite: 205, 206].

### Task 2: Closed-Set Speaker Authentication
To combat heavy data imbalance among different speakers, a **Balanced Sampling** strategy was employed to ensure fairness across all classes[cite: 207]. Models were evaluated across 3 randomized batches.
*   **SVM** and **MLP** provided the most stable and highest-performing results, consistently reaching up to **96.8% accuracy** in identifying the correct speaker[cite: 211, 213, 218].

---

## 🧩 Unsupervised Learning (Clustering)
To understand the natural grouping of the acoustic features without relying on labels, we utilized **PCA** and **t-SNE** to project the high-dimensional data into 2D and 3D latent spaces[cite: 191, 192, 222].

We evaluated two clustering algorithms for optimal cluster discovery:
1.  **K-Means++:** Using the **Silhouette Score**, we determined the optimal number of clusters to be **$k=2$** (aligning perfectly with the binary gender distribution in the dataset)[cite: 221].
2.  **Gaussian Mixture Models (GMM):** Because GMM models probability distributions rather than hard Euclidean boundaries, it successfully identified complex, non-spherical clusters. The Silhouette analysis for GMM also converged on **$k=2$** as the optimal structure[cite: 230, 232, 235].

*3D PCA Latent Space Visualization for GMM Clustering ($k=2$):*
<br>
![3D Clustering Visualization](assets/clustering_3d_visuals.png)

> **Insight on Over-segmentation:** Testing higher cluster counts ($k=4$ and $k=20$) resulted in Silhouette scores dropping significantly (even below zero for GMM), proving that forcing too many clusters leads to over-segmentation and artificial boundaries[cite: 232, 239].

---

## 📈 Key Results Summary

| Task | Best Model (Accuracy) | Best Model (Robustness/AUC) | Key Challenge Addressed |
| :--- | :--- | :--- | :--- |
| **Gender Classification** | MLP (93.8%) | XGBoost (AUC: 0.68) | Feature Extraction & Denoising |
| **Speaker Authentication** | SVM (96.8%) | MLP (95.8%) | Class Imbalance (Balanced Sampling) |
| **Acoustic Clustering** | GMM ($k=2$) | KMeans++ ($k=2$) | Dimensionality Reduction (PCA/t-SNE) |

---

````
voice-authentication-ml-pipeline/
├── assets/                             # Output visualizations and graphs
├── data/                               
│   ├── raw/                            # Raw .mp3 files (ignored in git)
│   └── processed/                      # Output features (features.csv)
├── docs/                               # Project reports and PDF documentation
├── notebooks/                          
│   └── final_project.ipynb             # Main execution notebook
├── src/                                # Modularized Python scripts
│   ├── audio_processor.py              # DSP and noise reduction logic
│   ├── feature_extractor.py            # Feature extraction (MFCC, etc.)
│   ├── classifiers.py                  # Supervised models (SVM, MLP, etc.)
│   └── clustering.py                   # Unsupervised models (GMM, KMeans)
├── requirements.txt                    # Project dependencies
└── README.md
````

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/voice-authentication-ml-pipeline.git](https://github.com/yourusername/voice-authentication-ml-pipeline.git)
   cd voice-authentication-ml-pipeline
