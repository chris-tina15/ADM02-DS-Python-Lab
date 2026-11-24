
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Generate house dataset
np.random.seed(10)
n = 150

area = np.random.randint(500, 3000, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 30, n)
distance = np.round(np.random.uniform(1, 25, n), 1)

price = (
    15
    + 0.045 * area
    + 4 * bedrooms
    - 0.6 * age
    - 1.2 * distance
    + np.random.normal(0, 6, n)
)

houses = pd.DataFrame({
    "area_sqft": area,
    "bedrooms": bedrooms,
    "age_years": age,
    "distance_km": distance,
    "price_lakhs": price.round(2)
})

print("House Price Prediction using Linear Regression")
print("=" * 50)

print("\nFirst five records:")
print(houses.head().to_string(index=False))

# Step 2: Correlation
print("\nCorrelation with price:")
print(
    houses.corr()["price_lakhs"]
    .drop("price_lakhs")
    .round(3)
    .to_string()
)

# Step 3: Split dataset
X = houses.drop(columns="price_lakhs")
y = houses["price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Step 4: Simple Linear Regression
simple = LinearRegression().fit(
    X_train[["area_sqft"]],
    y_train
)

print("\nSimple Linear Regression: price = b0 + b1 * area")

print(f"Intercept = {simple.intercept_:.4f}")
print(f"Slope = {simple.coef_[0]:.4f}")

simple_pred = simple.predict(X_test[["area_sqft"]])

print(f"R2 on test = {r2_score(y_test, simple_pred):.4f}")

# Step 5: Multiple Linear Regression
model = LinearRegression().fit(X_train, y_train)

print("\nMultiple Linear Regression coefficients:")
print(f"Intercept: {model.intercept_:.4f}")

for name, c in zip(X.columns, model.coef_):
    print(f"{name}: {c:.4f}")

# Step 6: Model evaluation
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)

print("\nModel evaluation on test set:")
print(f"MAE = {mean_absolute_error(y_test, pred):.4f}")
print(f"MSE = {mse:.4f}")
print(f"RMSE = {np.sqrt(mse):.4f}")
print(f"R2 = {r2_score(y_test, pred):.4f}")

# Step 7: Predict a new house
new_house = pd.DataFrame(
    [[1800, 3, 5, 8.0]],
    columns=X.columns
)

print(
    "\nPredicted price for 1800 sqft, 3 BHK, 5 years old, 8 km away:",
    f"{model.predict(new_house)[0]:.2f} lakhs"
)

# Step 8: Visualisation
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

xs = np.linspace(500, 3000, 100).reshape(-1, 1)

ax[0].scatter(
    X_train["area_sqft"],
    y_train,
    s=15,
    label="Training data"
)

ax[0].plot(
    xs,
    simple.predict(
        pd.DataFrame(xs, columns=["area_sqft"])
    ),
    color="red",
    label="Fitted line"
)

ax[0].set_xlabel("Area (sqft)")
ax[0].set_ylabel("Price (lakhs)")
ax[0].set_title("Simple Linear Regression")
ax[0].legend()

# Residual plot
ax[1].scatter(pred, y_test - pred, color="purple")
ax[1].axhline(0, color="black", linestyle="--")

ax[1].set_xlabel("Predicted price")
ax[1].set_ylabel("Residual")
ax[1].set_title("Residual Plot (Multiple Regression)")

plt.tight_layout()

plt.savefig("Experiment_05/exp05_regression.png", dpi=150)

plt.show()

print("\nPlot saved as exp05_regression.png")
print("\nEXPERIMENT 5 COMPLETED SUCCESSFULLY!")
