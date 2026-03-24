# 🛒 Customer Segmentation App

A machine learning project that segments customers into behavioral groups using **K-Means Clustering**, with an interactive **Streamlit** web app for real-time prediction.

---

## 📌 Project Overview

This project analyzes customer data to identify distinct segments based on demographics, spending behavior, and purchase patterns. It uses unsupervised learning (K-Means) to group customers into 6 clusters, helping businesses personalize marketing strategies.

---

## 🚀 Live Demo

Run locally with:
```bash
streamlit run segmentation.py
```

---

## 📁 Project Structure

```
Customer-Segmentation/
│
├── Analysis_Model.ipynb        # EDA + model training notebook
├── segmentation.py             # Streamlit web app
├── customer_segmentation.csv   # Raw dataset
├── kmeans_mode.pkl             # Trained K-Means model
├── scaler.pkl                  # Fitted StandardScaler
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🧠 ML Pipeline

1. **Data Cleaning** — dropped nulls, parsed dates
2. **Feature Engineering**
   - `Age` = 2025 − Year_Birth
   - `Total_Spending` = sum of all product spend columns
   - `Total_Children` = Kidhome + Teenhome
   - `Customer_Since` = days since joining
3. **EDA** — distributions, correlation heatmap, group analysis by education/marital status/age group
4. **Clustering** — StandardScaler → Elbow Method → K-Means (k=6)
5. **Visualization** — PCA 2D scatter plot of clusters

---

## 📊 Features Used for Clustering

| Feature | Description |
|---|---|
| Age | Customer age |
| Income | Annual income |
| Total_Spending | Sum of all product purchases |
| NumWebPurchases | Online purchases count |
| NumStorePurchases | In-store purchases count |
| NumWebVisitsMonth | Monthly website visits |
| Recency | Days since last purchase |

---

## 🖥️ Web App

The Streamlit app accepts customer details as input and predicts which cluster they belong to.

**Input fields:**
- Age, Income, Total Spending
- Web Purchases, Store Purchases, Web Visits/Month
- Recency (days since last purchase)

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run segmentation.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib
seaborn
```

> Full pinned versions in `requirements.txt`

---

## 📈 Results

- **6 customer clusters** identified via Elbow Method
- Clusters vary significantly in income level, spending habits, and purchase channel preferences
- PCA visualization confirms well-separated groupings

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-data-lightblue?logo=pandas)

---

## 👤 Author

**Nipun Bansal**  
📧 nipunbansal01@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/nipun-bansal-b20b9b269/) | [GitHub](https://github.com/Nipun0101)

---
