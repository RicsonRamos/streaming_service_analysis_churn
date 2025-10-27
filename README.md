# 🚀 Predictive Churn Analysis (Optimized XGBoost Model)

## 📝 Project Overview

This project implements a comprehensive Data Science pipeline focused on **predicting customer Churn (attrition)** within a telecommunications or subscription service context. The primary goal is to identify customers at the highest risk of cancellation (*churners*) before they terminate their service, enabling the company to deploy targeted and timely retention strategies.

The analysis compares a baseline model (Logistic Regression) against a high-performance, optimized model (XGBoost), prioritizing the **ROC AUC** metric to effectively handle the class imbalance inherent in most churn prediction problems.

---

## 🎯 Objectives

1.  **Exploratory Data Analysis (EDA) & Cleaning:** Identify and treat missing values, and perform necessary Feature Engineering.
2.  **Preprocessing:** Apply *One-Hot Encoding* to categorical variables and scaling (standardization) to numerical features.
3.  **Comparative Modeling:** Train **Logistic Regression (Baseline)** and the advanced **XGBoost Classifier**.
4.  **Optimization:** Utilize `RandomizedSearchCV` to fine-tune the XGBoost hyperparameters, specifically aiming to maximize the **ROC AUC score**.
5.  **Interpretation:** Analyze **Feature Importance** to understand which factors (e.g., contract length, services used, monthly spend) are the strongest drivers of churn.

---

## ⚙️ Technologies and Libraries

| Category | Library/Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python | Primary programming language. |
| **Data Handling** | Pandas, NumPy | Data processing, cleaning, and feature engineering. |
| **Baseline Model** | Scikit-learn (Logistic Regression) | Simple, interpretable model used as a performance benchmark. |
| **Advanced Model** | XGBoost (XGBClassifier) | High-performance Gradient Boosting algorithm suitable for complex classification. |
| **Optimization** | Scikit-learn (RandomizedSearchCV) | Efficient hyperparameter tuning to find optimal model settings. |
| **Metrics** | Scikit-learn (ROC AUC, Classification Report) | Performance evaluation, crucial for imbalanced datasets. |

---

## 🛠️ Methodology and ML Pipeline

The Machine Learning pipeline included several critical steps to ensure a robust and generalized model:

1.  **Missing Value Imputation:** Missing values (likely identified in the monthly charges column) were imputed, often treated as zero if absence indicates no service/charge.
2.  **Feature Encoding & Scaling:** All features were prepared for modeling using One-Hot Encoding and appropriate scaling.
3.  **Handling Class Imbalance:** The `scale_pos_weight` parameter in XGBoost was used to penalize misclassifications of the minority class (Churn), preventing the model from becoming biased toward the majority class (Non-Churn).
4.  **Robust Optimization:** **`RandomizedSearchCV`** was employed for efficient exploration of the hyperparameter space, leading to the best possible configuration for the XGBoost model.
5.  **Validation:** The **ROC AUC** metric was chosen as the main evaluator, as it accurately measures the model's ability to discriminate between the Churn and Non-Churn classes across all probability thresholds.

---

## 💡 Key Results and Insights

The model achieved the following key results after optimization and careful feature selection:

* **Final Optimized XGBoost Performance:** The final model achieved a robust **ROC AUC of 0.8549**. This score validates the model's high predictive power, demonstrating its ability to accurately rank customers by their risk of churning.
* **Feature Leakage Mitigation:** The pipeline successfully identified and handled a potential feature leakage (evidenced by an initial ROC AUC near 1.00), ensuring that the final performance is generalized and reliable.
* **Key Churn Drivers:** Feature Importance analysis highlighted that **Contract Duration (Tenure)** and **Type of Service (e.g., Internet Service)** are the most critical factors driving churn. This insight directs the business to prioritize improving specific service quality and promoting long-term contract renewals.

---

## 📂 Repository Structure

. ├── predictive-analysis-churn-model-xgboost.ipynb # Main project notebook ├── README.md # This file └── data/ └── [DATASET_NAME].csv # Original source data file

---

## 🚀 How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_LINK]
    cd [REPOSITORY_NAME]
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```

3.  **Execute the Notebook:**
    Open the file `predictive-analysis-churn-model-xgboost.ipynb` in your preferred environment
