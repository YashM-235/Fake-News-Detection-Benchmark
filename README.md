# 📰 Fake News Detection Benchmark

### ML & Deep Learning Comparative Study Across Multiple Datasets

---

## 📌 Overview

This project presents a **comprehensive benchmark study** on fake news detection using a wide range of **Machine Learning (ML)** and **Deep Learning (DL)** models.

Unlike traditional approaches that rely on a single dataset or model, this project focuses on **cross-dataset evaluation** to determine which models perform best under different data distributions.

---

## 🎯 Objectives

* 📊 Compare **17+ ML & DL models**
* 🧠 Analyze performance across **3 different datasets**
* 📈 Evaluate using **multiple performance metrics**
* 🔍 Identify the **best model for each dataset**
* ⚖️ Study **generalization vs dataset dependency**

---

## 📂 Datasets

* **Real and Fake News Dataset:**
  [https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset](https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset)

* **India Headlines News Dataset:**
  [https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset)

* **Global News Dataset:**
  [https://www.kaggle.com/datasets/everydaycodings/global-news-dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset)

---

## 🧠 Models Implemented

### 🔹 Machine Learning

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* Extra Trees
* Gradient Boosting
* AdaBoost
* Ridge Classifier
* XGBoost
* LightGBM

### 🔹 Deep Learning

* TextCNN
* LSTM
* Bidirectional LSTM
* GRU
* Bidirectional GRU
* CNN-LSTM Hybrid
* Deep MLP (Word2Vec)

---

📊 Evaluation Metrics

The models are evaluated using a comprehensive set of classification and regression-based metrics:

Accuracy – Overall correctness of predictions
Precision – Proportion of correct positive predictions
Recall – Ability to capture actual positive cases
F1-Score – Harmonic mean of Precision and Recall
ROC-AUC – Ability to distinguish between classes

🔹 Error & Statistical Metrics
RMSE (Root Mean Squared Error)
MSE (Mean Squared Error)
MAE (Mean Absolute Error)
R² Score (Coefficient of Determination)
MAPE (%) (Mean Absolute Percentage Error)

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Text cleaning (lowercasing, punctuation removal, stopwords)
* Tokenization
* Feature extraction using **TF-IDF** and **Word2Vec**

### 2. Model Training

* ML models trained on TF-IDF features
* DL models trained on sequence embeddings

### 3. Hyperparameter Tuning

* Grid Search & manual tuning

### 4. Evaluation

* Cross-dataset comparison
* Confusion matrix & classification reports
* Heatmaps and pairplots for deeper insights

---

## 📈 Key Insights

* 🚫 No single model dominates all datasets
* 🌲 Ensemble models (Random Forest, Gradient Boosting) show strong consistency
* 🧠 Deep Learning models perform better on larger datasets
* ⚠️ Model performance is highly dataset-dependent

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* TensorFlow / Keras / PyTorch
* XGBoost, LightGBM
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/Fake-News-Detection-Benchmark.git

# Navigate into project
cd Fake-News-Detection-Benchmark

# Install dependencies
pip install -r requirements.txt

# Run notebooks or scripts
```

---

## 🔮 Future Work

* 🤖 Transformer models (BERT, RoBERTa)
* 🌐 Real-time fake news detection system
* 📊 Explainable AI (LIME, SHAP)
* 🚀 Deployment using Streamlit / Flask

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is intended for **academic and research purposes**.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---
