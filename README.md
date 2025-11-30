# ML-Classification and Customer Churn

![Machine Learning](images/ml_classification.png)

## 📖 Overview

This repository contains projects and exercises on classification algorithms and customer churn prediction with animations and visualizations. The goal is to learn, implement, and evaluate machine learning models, including practical portfolio projects.

We cover:

* k-Nearest Neighbors (k-NN)
* Decision Tree
* Random Forest
* Logistic Regression
* Support Vector Machines (SVM)
* Customer churn prediction with ensemble methods (Random Forest, XGBoost)

---

## 📚 17. Classification

### Topics:

1. 17.1 Kirish
2. 17.2 k-NN
3. 17.3 Classificatorʼni baholash: Jaccard Index
4. 17.4 Classificatorʼni baholash: Confusion Matrix (Precision, Recall)
5. 17.5 k-NN implementation with scikit-learn
6. 17.6 Selecting the best k for k-NN
7. 17.7 Jupyter Notebook: k-NN
8. 17.8 Portfolio project: Diabetes diagnosis
9. 17.9 Decision Tree algorithm
10. 17.10 Decision Tree with scikit-learn
11. 17.11 Decision Tree visualization
12. 17.12 Decision Tree hyperparameters
13. 17.13 Random Forest
14. 17.14 Jupyter Notebook: Decision Tree
15. 17.15 Logistic Regression
16. 17.16 Choosing the best model
17. 17.17 Final classification exercise

### Example Animations

![k-NN animation](images/knn_animation.gif)
![Decision Tree Growth](images/decision_tree_grow.gif)

---

## 📊 18. Customer Churn Prediction

### Topics:

1. 18.1 What is Customer Churn?
2. 18.2 Customer Churn: Data Analysis
3. 18.3 Logistic Regression, SVM, ROC Curve
4. 18.4 Decision Tree, Random Forest, XGBoost
5. 18.5 Jupyter Notebook: Customer Churn
6. 18.6 Final practice: Customer Churn prediction

### Overview

Customer churn refers to customers stopping use of a service or product. Predicting churn allows businesses to retain valuable clients. Models include Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost.

### Example Animations

![ROC Curve](images/roc_curve.gif)
![Random Forest Feature Importance](images/rf_feature_importance.gif)

---

## 🔧 Data Preprocessing

* Handle missing values (drop or impute)
* Encode categorical variables using pd.get_dummies()
* Scale features with StandardScaler()
* Split data into training and test sets (train_test_split)

---

## 🤖 Models Included

* k-NN
* Decision Tree
* Random Forest
* Logistic Regression
* Support Vector Machines (SVM)
* XGBoost (for Customer Churn)

Evaluation metrics:

* Accuracy
* Precision / Recall
* Jaccard Index
* Confusion Matrix
* ROC Curve

---

## 📂 Repository Structure

ML-Classification-and-Customer-Churn/
│
├── notebooks/
│   ├── kNN.ipynb
│   ├── DecisionTree.ipynb
│   ├── RandomForest.ipynb
│   └── CustomerChurn.ipynb
│
├── data/
│   └── E-Commerce-Dataset.xlsx
│
├── images/
│   ├── ml_classification.png
│   ├── knn_animation.gif
│   ├── decision_tree_grow.gif
│   ├── churn_pie.png
│   ├── tenure_hist.png
