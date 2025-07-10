import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # You can use LogisticRegression or others
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load your dataset
df = pd.read_csv('loan_data.csv')  # Replace with your file path

# 2. Prepare features and target
X = df[['Age', 'LoanAmount']]
y = df['LoanSanctioned']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Predict for a new customer
def predict_loan(age, loan_amount):
    return model.predict([[age, loan_amount]])[0]

# Example usage:
result = predict_loan(30, 100000)
print("Loan Sanctioned" if result == 1 else "Loan Not Sanctioned")

plt.figure(figsize=(8,6))
plt.scatter(X_test['Age'], X_test['LoanAmount'], c=y_pred, cmap='bwr', edgecolor='k', s=100)
plt.xlabel('Age')
plt.ylabel('Loan Amount')
plt.title('Test Data Predictions (Red=Not Sanctioned, Blue=Sanctioned)')
plt.show()
