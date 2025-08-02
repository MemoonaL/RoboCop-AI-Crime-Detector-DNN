# RoboCop-AI-Crime-Detector-DNN
Imagine a futuristic city run by AI. The government deploys a Robot Police Cop (powered by a Deep Neural Network) to predict whether a person will commit a crime. But… the robot starts failing , it overfits to past crime data and can’t generalize to new situations. Now we retrain it with regularization to make it smarter and fairer.
A deep learning-based project that predicts violent crime probability in U.S. communities using socio-economic, policing, and demographic data. Inspired by the concept of an AI-powered robot police officer ("RoboCop"), this project explores overfitting, regularization, and hyperparameter optimization using TensorFlow and Scikit-learn.

---

## 🧠 Project Objectives

- Train a deep neural network (DNN) to classify communities as **low or high crime risk**
- Predict **violent crime rate** (regression)
- Demonstrate and visualize **overfitting**
- Apply **Dropout** and **L2 Regularization** to improve generalization
- Use **GridSearchCV** for hyperparameter tuning

---

## 📁 Dataset

- **UCI Communities and Crime Dataset**  
  [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)

This dataset includes over 100 features related to:
- Demographics (age, race, income, education, etc.)
- Policing data
- Socioeconomic indicators
- Target: `ViolentCrimesPerPop` (continuous) + derived binary label `CrimeBinary`

---

## 📦 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy, Matplotlib

---

## 🚦 Model Versions

| Model | Description |
|-------|-------------|
| **RoboCop v1** | Simple DNN with no regularization (overfits easily) |
| **RoboCop v2** | DNN with Dropout and L2 Regularization |
| **RoboCop v3** | DNN optimized using GridSearchCV (dropout rate & optimizer tuning) |

---

## 🧪 Performance Metrics

- Classification: Accuracy, F1-score, Validation Curves
- Regression: MSE, R²
- Overfitting visualized using train vs validation loss plots

---
