# Credit Card Default Prediction

## 📘 Overview
This project implements machine learning models to predict whether a credit card client will default on their next payment.  
It was developed as part of coursework for the University of Huddersfield, Data Forensics / Machine Learning module.

Two models are included for comparison:
1. **Decision Tree Classifier** – Simple and interpretable, but can overfit.
2. **Random Forest Classifier** – Ensemble of decision trees, better generalization and higher accuracy.

---

## 📊 Dataset
The project uses the **Default of Credit Card Clients Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients), also available on [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset).

**Features include:**
- Demographics: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- Credit limit: `LIMIT_BAL`
- Bill and payment amounts: `BILL_AMT1–BILL_AMT6`, `PAY_AMT1–PAY_AMT6`
- Repayment status: `PAY_0–PAY_6`

**Target variable:**  
`default_payment_next_month`  
- 0 = No default  
- 1 = Default  

> ⚠️ The dataset itself is **not included** in this repo. Users should download it from Kaggle or UCI.

---

## 🧹 Data Preprocessing
- Missing values are filled with the column mean.  
- Outliers in `LIMIT_BAL` are removed using Z-score > 5.  
- Categorical variables (`SEX`, `EDUCATION`, `MARRIAGE`) are encoded with `LabelEncoder`.  
- Numerical features are scaled with `StandardScaler`.

---

## 🧠 Model Training
Scripts:
- `src/train_decision_tree.py` → trains and evaluates a Decision Tree classifier.  
- `src/train_random_forest.py` → trains and evaluates a Random Forest classifier.  

Each script:
1. Splits data into training (70%) and testing (30%).  
2. Trains the model.  
3. Evaluates accuracy, confusion matrix, ROC curve, feature importance, and error metrics.  
4. Saves the trained model in `model/` as `.pkl`.

---

## 🧪 Model Comparison
Metrics and visualizations allow comparing the performance of the two models:
- Accuracy and classification report  
- Confusion matrix  
- ROC curve and AUC  
- Feature importance  

Evaluation scripts are provided in `src/evaluate_models.py`.

---

## 📦 Predicting New Data
`src/predict_new_data.py` can be used to predict default probability for a new client.  
- Ensure the input has the same columns as the training data.  
- The script applies the same scaling and encoding as the training process.

---

## 🛠️ Requirements
- Python 3.9+  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn  
- Matplotlib  
- SciPy  
- Joblib  
- KaggleHub (optional, for dataset download)

Install with:

```bash
pip install -r requirements.txt
