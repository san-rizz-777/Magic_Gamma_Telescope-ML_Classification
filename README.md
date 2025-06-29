# 🌌 MAGIC Gamma Telescope - ML Classification

This project focuses on classifying high-energy gamma particles versus hadronic background using data from the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). We apply multiple Machine Learning algorithms and evaluate their performance using precision, recall, F1-score, and accuracy metrics.

[🔗 Dataset on UCI Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)  


---

## 📂 Dataset Overview

- **Source**: UCI Machine Learning Repository
- **Samples**: 19,020
- **Features**: 10 numerical attributes describing gamma-ray and hadron events
- **Target classes**:
  - `g` → gamma ray (signal, labeled as 1)
  - `h` → hadron (background noise, labeled as 0)

---

## 🧠 Applied ML Models

We evaluated the following classification models:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayes**
4. **Support Vector Machine (SVM)**
5.  **Neural Network (Keras)**


---

## 📊 Model Performance Summary

| Model               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|---------------------|----------|----------------------|------------------|---------------------|
| KNNs                | 0.79     | 0.85                 | 0.83             | 0.84                |
| Naive - Bayes       | 0.73     | 0.75                 | 0.88             | 0.81                |
| Logistic Regression | 0.77     | 0.83                 | 0.81             | 0.82                |
| SVM                 | 0.85     | 0.88                 | 0.88             | 0.88                |
| Neural Network      | **0.87** | **0.88**             | **0.93**         | **0.90**            |

---

## 📈 Why SVM and Neural Network Performed Better

### ✅ **Support Vector Machine (SVM)**
- SVM works well with high-dimensional data and clear margins of separation, both of which are present in this dataset.
- The dataset was preprocessed and scaled, which is ideal for SVM.
- It effectively finds the optimal decision boundary between gamma and hadron events.

### ✅ **Neural Network**
- The feed-forward Keras model was able to capture non-linear relationships and interactions between features.
- Class imbalance was handled via oversampling, improving generalization.
- The model benefited from large training data and feature normalization, improving convergence and accuracy.

---

## ⚙️ Project Workflow

1. **Data Cleaning & Preprocessing**
   - Label encoding
   - Standardization
   - Train/validation/test split (60/20/20)

2. **Balancing**
   - Random oversampling applied to training data to address class imbalance

3. **Model Training**
   - Each model trained and validated separately
   - Hyperparameters chosen manually for simplicity

4. **Evaluation**
   - Used `classification_report` (precision, recall, f1-score)
   - Plotted histogram distribution of features by class

---

## 📁 Project Files

- `Project_1_ml.ipynb`: Full code with preprocessing, training, and evaluation
- `*.png`: Feature-wise histograms for visual understanding
- `README.md`: Project documentation

---

## 🧠 Future Work

- Hyperparameter tuning (GridSearchCV or Optuna)
- Deep learning with more layers and dropout
- Use of techniques like PCA for dimensionality reduction
- Visualization with t-SNE or UMAP for cluster insights

---

## 🤝 Acknowledgements

- MAGIC Collaboration and UCI ML Repository for the dataset
- Scikit-learn, Keras libraries for modeling

---


