# 🛡️ Streaming Service Churn Radar  
### An End-to-End Data Science Project with Business Impact

This project showcases an **end-to-end data science solution** designed to **reduce customer churn and protect recurring revenue** in a subscription-based streaming service.

It demonstrates not only model development, but also **product thinking**, **business metrics**, and **decision support**, bridging the gap between data science and real-world impact.

---

## 🎯 Business Problem

Customer churn is one of the main drivers of revenue loss in subscription businesses.

The goal of this project is to:
- Identify customers at high risk of cancellation
- Quantify the financial impact of churn
- Support marketing and retention teams with **actionable insights**, not just predictions

---

## 💡 Solution Overview

The solution uses a **machine learning model (XGBoost)** to assign a **churn risk score** to each customer, combined with a **business dashboard** that helps prioritize retention actions.

Key characteristics:
- Probabilistic churn prediction (not only yes/no)
- Customer prioritization based on **risk × lifetime value (LTV)**
- Clear connection between model outputs and business decisions

---

## 🧠 Data Science Approach

- Exploratory Data Analysis (EDA) to identify churn drivers
- Feature engineering and preprocessing using a reproducible pipeline
- Supervised learning with XGBoost for tabular data
- Proper handling of class imbalance
- Model outputs designed for business consumption

---

## 📊 Decision Support Dashboard (Streamlit)

The interactive dashboard transforms model predictions into **management-ready insights**:

- **Churn Risk Segmentation:** Low, Medium, and High risk customers
- **Retention ROI Simulator:** Estimate the financial return of retention strategies
- **Customer Prioritization Matrix:** Focus efforts where they matter most
- **Exportable Lead List:** High-risk customers ready for action by marketing or sales teams

---

## 📈 Key Business Insights

Insights uncovered during analysis include:

- Customers with frequent support requests are significantly more likely to churn
- Low engagement is a strong early warning signal
- Price increases without perceived value upgrades drive churn in specific regions

These insights can directly inform **retention campaigns, pricing strategy, and customer experience improvements**.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-Learn**
- **XGBoost**
- **Streamlit**
- **Joblib**

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py

👤 About the Author

Ricson Ramos
Data Scientist | Machine Learning | Business Analytics  
