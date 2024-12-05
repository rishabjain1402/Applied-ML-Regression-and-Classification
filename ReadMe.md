### GitHub Repository Name:
**Applied-ML-Regression-and-Classification**

### README Description:

# Applied Machine Learning: Regression and Classification

This repository contains **Assignment 1** for **ORIE 5750: Applied Machine Learning**, where we explore foundational concepts in regression and classification through real-world datasets. The assignment emphasizes data preprocessing, feature engineering, and implementing machine learning models.

---

## Project Overview

This project is divided into two main parts:

### **Part 1: House Price Prediction**
We predict house sale prices using regression techniques, focusing on:
- **Data Preprocessing**:
  - Handling missing values via imputation (mean/mode).
  - Dropping irrelevant or sparse columns.
  - Normalizing numerical features for improved model performance.
- **Feature Engineering**:
  - Selecting features highly correlated with the target variable (`SalePrice`).
  - Applying one-hot encoding to categorical variables.
- **Modeling**:
  - Ordinary Least Squares (OLS) regression.
  - Evaluating performance using R-squared and Mean Squared Error (MSE).

### **Part 2: Titanic Survival Classification**
We predict survival probabilities of passengers from the Titanic dataset using logistic regression:
- **Data Cleaning**:
  - Dropping irrelevant features (e.g., `Name`, `Ticket`, `Cabin`).
  - Imputing missing values for `Age`, `Fare`, and `Embarked`.
- **Feature Transformation**:
  - Mapping categorical variables (e.g., `Sex` to binary values).
  - One-hot encoding of `Embarked`.
- **Modeling**:
  - Logistic regression with scaled features.
  - Evaluating model accuracy and survival predictions.

---

## Key Results
- **House Price Prediction**:
  - Achieved an R-squared value of **0.774**, indicating strong predictive performance.
- **Titanic Survival Classification**:
  - Predicted survival probabilities for unseen test data, with approximately **4% of test passengers** classified as survivors.

---

## Technologies Used
- **Python**:
  - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- **Machine Learning**:
  - Ordinary Least Squares Regression
  - Logistic Regression
- **Data Visualization**:
  - Histograms and bar plots using `matplotlib` and `seaborn`.

---

## Repository Contents
- **`Assignment_1_Write_Up.pdf`**: Detailed write-up of the assignment.
- **`house_price_prediction.ipynb`**: Jupyter Notebook for Part 1 (House Price Prediction).
- **`titanic_survival_classification.ipynb`**: Jupyter Notebook for Part 2 (Titanic Survival Classification).
- **`data/`**: Folder containing training and testing datasets.

---

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/Applied-ML-Regression-and-Classification.git
   cd Applied-ML-Regression-and-Classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn sklearn
   ```

3. **Run the Notebooks**:
   - Open and execute `house_price_prediction.ipynb` for Part 1.
   - Open and execute `titanic_survival_classification.ipynb` for Part 2.

---

## Authors
- **Rishab Jain** (`rj424`)
- **Shalom Otieno** (`soo26`)
