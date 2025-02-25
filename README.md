# 📊 Linear Regression – Car Price Prediction

## 🌟 Highlights
- **Machine Learning from Scratch** – No Scikit-Learn or built-in regression functions.
- **Gradient Descent Optimization** – Supports multiple learning rates to optimize training.
- **Data Normalization & Feature Scaling** – Ensures stable convergence during training.
- **Model Persistence** – Saves trained parameters for future predictions.
- **Visualization** – Plots regression results and dataset for better interpretation.
- **Command-Line Interface** – Train and predict using simple CLI commands.

---

## ℹ️ Overview
**This project implements a simple linear regression model** to predict car prices based on mileage. Using a gradient descent algorithm, the model optimizes **theta0 (intercept) and theta1 (coefficient)** to fit a linear function of the form:

```
estimatePrice(mileage) = theta0 + (theta1 * mileage)
```

**Key Features:**
✅ Implements **gradient descent** to iteratively improve accuracy.
✅ **Normalizes features** for better convergence.
✅ **Saves trained model parameters** in `model.json` for future predictions.
✅ **Provides visualization tools** to analyze dataset distribution and model fit.
✅ Supports **command-line input for training and predictions**.

---

## 🚀 Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/LuckyIntegral/linear_regression.git
cd linear_regression
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔄 Training the Model
Run the training script to optimize model parameters:
```bash
python linear_regression/train.py
```
This will generate a **model file (`model.json`)** storing the trained values of theta0 and theta1.

---

## 🔮 Making Predictions
Use the trained model to predict the price of a car based on mileage:
```bash
python linear_regression/predict.py
```
This script prompts the user for mileage input and returns an estimated car price.

---

## 📊 Data Visualization
Before training, analyze the dataset using built-in visualization tools:

### **Generate a Regression Plot**
```bash
python linear_regression/predict.py
```
This will display a graph of the dataset, the regression line, and the user’s prediction.

---

## 🧪 Project Structure
```
.
├── README.md
├── data
│   └── data.csv
├── linear_regression
│   ├── predict.py
│   └── train.py
└── requirements.txt
```

---

## ✨ Future Improvements
- Add **Polynomial Regression for non-linear relationships**.
- Support **Multiple Linear Regression with additional features**.
- Improve **Precision Score Calculation**.

---

## 🎓 Author
**Vitalii Frants**
📍 42 Vienna – AI & Algorithms
👉 [GitHub](https://github.com/LuckyIntegral)

---

### **💎 Ready to Predict Car Prices? Try it now!**

