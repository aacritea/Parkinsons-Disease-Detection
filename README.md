# Parkinson-s-Disease

# 🧠 Parkinson's Disease Detection using Voice Analysis

This project focuses on **early detection of Parkinson’s Disease (PD)** using **machine learning on vocal biomarkers**.  
The model analyses voice features such as jitter, shimmer, and frequency variation to classify whether a patient has Parkinson’s or not — achieving **92.3% accuracy** and an **ROC-AUC of 0.96**.

---

## 📊 Project Overview

| Category | Details |
|-----------|----------|
| **Domain** | Healthcare / Bioinformatics / AI |
| **Objective** | Early diagnosis of Parkinson’s disease using voice measurements |
| **Algorithm Used** | Random Forest Classifier |
| **Accuracy** | 92.3% |
| **ROC-AUC Score** | 0.962 |
| **Dataset** | [UCI Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) |
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib |

---

## 🧩 Methodology

1. **Data Collection**  
   The dataset was obtained from the UCI Machine Learning Repository. It consists of 195 voice recordings, each with 23 biomedical voice measures.  

2. **Data Preprocessing**  
   - Dropped non-numeric columns (like patient name).  
   - Scaled features using `StandardScaler`.  

3. **Model Development**  
   - Used `RandomForestClassifier` from scikit-learn.  
   - Split data into 80% training and 20% testing.  
   - Trained and tuned hyperparameters for optimal performance.  

4. **Evaluation Metrics**  
   - Accuracy  
   - Precision, Recall, F1-Score  
   - ROC-AUC Curve  
   - Confusion Matrix  

---

## 📈 Results

| Metric | Score |
|---------|--------|
| **Accuracy** | 0.923 |
| **Precision (PD)** | 0.93 |
| **Recall (PD)** | 0.97

### 🔹 Confusion Matrix
[[ 8 2]
[ 1 28]]


### 🔹 Classification Report
| Class | Precision | Recall | F1-score |
|--------|------------|--------|-----------|
| 0 (Healthy) | 0.89 | 0.80 | 0.84 |
| 1 (Parkinson’s) | 0.93 | 0.97 | 0.95 |

---

## 🧠 Insights

- Voice-based biomarkers are a **non-invasive** and **low-cost** diagnostic tool.  
- Random Forest outperformed linear models in accuracy and robustness.  
- The model demonstrates potential for integration into **telemedicine platforms** or **mobile diagnostic tools**.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Parkinson-s-Disease.git
   cd Parkinson-s-Disease
   
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib

3. Run the project:
   ```bash
   python parkinsons_diagnosis.py

## 🧾 Future Work
- Experiment with Deep Learning models (LSTM, CNN) for audio feature extraction.
- Integrate the model with Streamlit for a web-based diagnostic tool.
- Perform feature importance analysis for interpretability.

## 📚 References
- Little, Max A., et al. "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection." BioMedical Engineering OnLine, 2007.
- UCI Parkinson’s Dataset
- [Scikit-learn Documentation](https://archive.ics.uci.edu/ml/datasets/parkinsons?utm_source=chatgpt.com)

## 👩‍💻 Authors
- Aakriti Jain, Ujjawal Gaur
- B.Tech in Artificial Intelligence and Data Science
- GGSIPU, 2026

If you found this project interesting, give it a star on GitHub! 🌟
