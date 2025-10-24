# 🏦 Sprint 10 Project — Beta Bank Customer Churn Prediction (Machine Learning)

---

🧠 **Project Overview**  
In this sprint, I worked as a data scientist for **Beta Bank**, a financial institution that noticed a growing customer churn problem.  
Since retaining existing customers is more cost-effective than acquiring new ones, the goal was to build a **machine learning model** that predicts whether a customer is likely to leave the bank soon.  

The task required achieving an **F1-score of at least 0.59** on the test set and comparing it against the **AUC-ROC** metric to assess classification quality.

This project consolidates all prior experience in data preprocessing, class imbalance correction, model optimization, and evaluation.

---

## 🎯 Project Objectives
- Load and prepare customer data (`/datasets/Churn.csv`).  
- Explore and clean the dataset (handle missing values, encoding, scaling).  
- Analyze class balance and visualize churn distribution.  
- Train baseline models **without addressing class imbalance** to establish reference scores.  
- Apply **two or more techniques** to handle class imbalance:
  - Class weighting (`class_weight='balanced'`)
  - Oversampling (e.g., `RandomOverSampler` or `SMOTE`)
  - Undersampling (e.g., `RandomUnderSampler`)
- Compare models and tune hyperparameters to maximize the **F1-score**.  
- Evaluate the best model on the **test set** and compute both **F1** and **AUC-ROC**.  

---

## 📁 Dataset Description
**File:** `Churn.csv`

Each record represents a **customer** of Beta Bank, including demographic data, financial information, and account activity.

| Column | Description |
|---------|--------------|
| `RowNumber` | Row index (not used in modeling) |
| `CustomerId` | Unique customer identifier |
| `Surname` | Customer surname |
| `CreditScore` | Credit rating |
| `Geography` | Country of residence |
| `Gender` | Gender (Male/Female) |
| `Age` | Customer age |
| `Tenure` | Years of bank relationship |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Credit card ownership (1 = Yes, 0 = No) |
| `IsActiveMember` | Activity status (1 = Active, 0 = Inactive) |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | Target variable — Churn indicator (1 = Customer left, 0 = Customer stayed) |

---

## 🧩 Project Steps

### Step 1 – Data Preparation  
- Loaded data and inspected structure with `.info()` and `.describe()`.  
- Checked for missing values and duplicates.  
- Encoded categorical features (`Geography`, `Gender`) using one-hot encoding.  
- Scaled numerical features using `StandardScaler`.  
- Split data into **train (60%)**, **validation (20%)**, and **test (20%)** sets.

### Step 2 – Baseline Model  
- Trained initial models (**Logistic Regression**, **Decision Tree**, **Random Forest**) without class balancing.  
- Measured F1 and AUC-ROC on validation data.  
- Observed poor recall due to imbalance (majority of “non-churn” cases).

### Step 3 – Handle Class Imbalance  
Applied and compared several techniques:
1. **Class Weighting:** adjusted algorithm sensitivity to minority class.  
2. **Oversampling:** increased churn cases using **SMOTE** or `RandomOverSampler`.  
3. **Undersampling:** reduced majority class samples to balance the dataset.  

Evaluated each technique with **cross-validation** and **grid search** for optimal hyperparameters (`max_depth`, `n_estimators`, `min_samples_split`).

### Step 4 – Model Evaluation  
- Selected the best-performing model based on **F1-score** and **AUC-ROC** on validation data.  
- Tested the final model on the **unseen test set**.  
- Verified that **F1 ≥ 0.59** to meet project requirements.  
- Compared **AUC-ROC** to ensure consistent performance across thresholds.  

### Step 5 – Results Interpretation  
- Analyzed feature importance (e.g., Age, Balance, Activity status).  
- Interpreted which factors most influence churn probability.  
- Discussed how Beta Bank can use these insights for customer retention strategies.

---

## 📊 Example Results (Illustrative)

| Model | F1 (Validation) | F1 (Test) | AUC-ROC (Test) |
|--------|------------------|------------|----------------|
| Logistic Regression | 0.52 | 0.51 | 0.78 |
| Decision Tree | 0.56 | 0.54 | 0.80 |
| **Random Forest + Class Weight** | **0.61** | **0.59** | **0.85** |
| **Random Forest + SMOTE** | **0.63** | **0.60** | **0.86** |

✅ **Final Model:** Random Forest with SMOTE oversampling  
✅ **F1 (test):** 0.60  
✅ **AUC-ROC (test):** 0.86  

---

## 💼 Skills Developed
- Data Preprocessing & Feature Encoding  
- Class Imbalance Handling (Class Weighting, SMOTE, Undersampling)  
- Model Selection & Hyperparameter Tuning  
- Evaluation Metrics (F1, ROC-AUC, Precision, Recall)  
- Business Insight Interpretation from ML Results  

---

## 🧰 Tools & Libraries
`Python` | `Pandas` | `NumPy` | `Matplotlib` | `Seaborn` | `scikit-learn` | `Imbalanced-learn` | `Jupyter Notebook`

---

## 👤 Author  
*Project completed by [Jonathan Peña] as part of Sprint 10 — Customer Churn Prediction with Machine Learning.*
