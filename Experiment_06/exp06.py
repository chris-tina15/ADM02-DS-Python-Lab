
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Step 1: Generate dataset
np.random.seed(3)
n = 400

tenure = np.random.randint(1, 72, n)
monthly = np.round(np.random.uniform(200, 1500, n), 0)
complaints = np.random.poisson(1.2, n)
contract = np.random.choice([0, 1], n, p=[0.55, 0.45])

# 1 = Yearly contract
logit = (
    -1.0
    - 0.06 * tenure
    + 0.002 * monthly
    + 0.7 * complaints
    - 1.3 * contract
)

churn = (
    np.random.rand(n) < 1 / (1 + np.exp(-logit))
).astype(int)

data = pd.DataFrame({
    "tenure_months": tenure,
    "monthly_bill": monthly,
    "complaints": complaints,
    "yearly_contract": contract,
    "churn": churn
})

print("Telecom Customer Churn Prediction using Logistic Regression")
print("=" * 60)

print(data.head().to_string(index=False))

print("\nChurn distribution:")
print(data["churn"].value_counts().to_dict())

# Step 2: Split dataset
X = data.drop(columns="churn")
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=0
)

# Step 3: Standardisation and Logistic Regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X_train, y_train)

lr = model.named_steps["logisticregression"]

print("\nCoefficients (standardised features):")

for name, c in zip(X.columns, lr.coef_[0]):
    print(
        f"{name:<16} {c:>8.4f} "
        f"odds ratio = {np.exp(c):.3f}"
    )

print(f"{'Intercept':<16} {lr.intercept_[0]:>8.4f}")

# Step 4: Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\nValidation on test set:")

print(f"Accuracy  = {accuracy_score(y_test, pred):.4f}")
print(f"Precision = {precision_score(y_test, pred):.4f}")
print(f"Recall    = {recall_score(y_test, pred):.4f}")
print(f"F1-score  = {f1_score(y_test, pred):.4f}")
print(f"ROC-AUC   = {roc_auc_score(y_test, prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Step 5: 5-fold cross-validation
cv = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n5-fold cross-validation accuracy:")
print(np.round(cv, 4))

print(f"Mean = {cv.mean():.4f}, Std = {cv.std():.4f}")

# Step 6: Predict new customer
customer = pd.DataFrame(
    [[4, 1250, 3, 0]],
    columns=X.columns
)

p = model.predict_proba(customer)[0, 1]

print(
    "\nNew customer (4 months, Rs.1250, "
    "3 complaints, monthly plan):"
)

print(f"Churn probability = {p:.3f}")

# Step 7: ROC curve
fpr, tpr, _ = roc_curve(y_test, prob)

plt.figure(figsize=(5.5, 4.5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression "
          f"(AUC = {roc_auc_score(y_test, prob):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Model")
plt.legend()
plt.tight_layout()

plt.savefig("Experiment_06/exp06_roc.png", dpi=150)
plt.show()

print("\nPlot saved as exp06_roc.png")
print("\nEXPERIMENT 6 COMPLETED SUCCESSFULLY!")
