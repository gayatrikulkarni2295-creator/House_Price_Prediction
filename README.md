<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=180&section=header&text=House%20Price%20Prediction&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Machine%20Learning%20%7C%20Random%20Forest%20%7C%20Python&descAlignY=58&descAlign=50" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

<br/>

**A Machine Learning project that predicts house prices based on features like area, bedrooms, location score, and more — achieving 93.6% accuracy using Random Forest Regression.**

</div>

---

## 📌 Overview

This project uses supervised Machine Learning to predict house prices. Two models are trained and compared — **Linear Regression** and **Random Forest Regressor** — with Random Forest achieving the best results.

---

## 🎯 Features Used for Prediction

| Feature | Description |
|---------|-------------|
| `area_sqft` | Total area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `house_age_years` | Age of the house in years |
| `garage` | Whether garage is available (0 or 1) |
| `location_score` | Location quality score (1–10) |

---

## 🧠 Models Trained

| Model | R² Score |
|-------|----------|
| Linear Regression | 98.17% |
| **Random Forest** ✅ | **93.6%** |

> Random Forest selected as final model due to better generalization and feature importance insights.

---

## 📈 Model Performance

```
MAE  (Mean Absolute Error)  →  ₹28,087
RMSE (Root Mean Square Error) →  ₹35,189
R²   (Accuracy Score)       →  93.6%
```

---

## 🎯 Feature Importance

```
area_sqft        ████████████████████████████████  77.7%
location_score   █████                             14.8%
house_age_years  █                                  4.9%
bedrooms                                            1.4%
garage                                              0.7%
bathrooms                                           0.6%
```

---

## 🏠 Sample Predictions

| Area | Bedrooms | Location Score | Predicted Price |
|------|----------|---------------|-----------------|
| 1200 sqft | 2 | 5/10 | ₹2,58,661 |
| 2500 sqft | 4 | 7/10 | ₹5,22,483 |
| 3500 sqft | 5 | 9/10 | ₹7,14,269 |

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── train_model.py          ← ML training script (main)
├── house_data.csv          ← Dataset (500 houses)
├── house_price_model.pkl   ← Trained model (generated after running)
├── requirements.txt        ← Dependencies
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/House-Price-Prediction.git
cd House-Price-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train_model.py
```

**4. Output**
```
Model trained with 93.6% accuracy
house_price_model.pkl saved ✅
```

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas** — Data loading and analysis
- **NumPy** — Numerical operations
- **Scikit-learn** — ML models (Linear Regression, Random Forest)
- **Pickle** — Model serialization

---

## 📄 License

MIT License — open source and free to use.

---

<div align="center">

*Built with Python & Scikit-learn*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=100&section=footer" width="100%"/>

</div>
