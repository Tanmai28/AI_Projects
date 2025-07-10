# 🏦 Loan Sanction Classification using Machine Learning

This project demonstrates a basic **classification** technique using input features like `Age` and `Loan Amount` to predict whether a **loan should be sanctioned** or not.

---

## ⚙️ Steps to Build the Loan Sanction Model

### 📥 Data Collection
A simple structured dataset containing:
- **Age** of the customer  
- **LoanAmount** requested  
- **LoanSanctioned** status (Yes=1 / No=0)

### 🧠 Feature Selection
- Input features: `Age`, `LoanAmount`  
- Output label: `LoanSanctioned` (binary classification)

### 🧪 Model Training
- Model used: **Random Forest Classifier**
- Trained on historical data to learn patterns of approval/rejection

### 🎯 Prediction
- Given a new customer with `Age` and `LoanAmount`, the model predicts if the **loan will be sanctioned or not**

---

## 📊 Visualization
The output includes a **scatter plot** that visually differentiates sanctioned vs non-sanctioned loans based on Age and Loan Amount.
![image](https://github.com/user-attachments/assets/127808df-daec-4332-b3c5-168990027ec3)


---

## Project Structure
```
loan-sanction-classifier/
├── loan_data.csv           # Sample dataset with Age, LoanAmount, LoanSanctioned
├── loan_classifier.py      # Main Python script: training, prediction, visualization
├── Output                 # Sample output and prediction plot
└── README.md               # Project documentation
```
