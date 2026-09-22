# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. LOAD DATASET
data = pd.read_csv("Salary_Data.csv")

print(data.head())
print(data.info())
print(data.describe())


# 3. VISUALIZE DATA
plt.scatter(data["YearsExperience"], data["Salary"])
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.show()


# 4. PREPARE DATA
X = data[["YearsExperience"]]
y = data["Salary"]


# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)


# 6. LINEAR REGRESSION
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)


# 7. LINEAR EVALUATION
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nLINEAR REGRESSION")
print("MAE:", mae_linear)
print("MSE:", mse_linear)
print("RMSE:", rmse_linear)
print("R2:", r2_linear)


# 8. POLYNOMIAL REGRESSION
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)


# 9. POLYNOMIAL EVALUATION
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPOLYNOMIAL REGRESSION")
print("MAE:", mae_poly)
print("MSE:", mse_poly)
print("RMSE:", rmse_poly)
print("R2:", r2_poly)


# 10. COMPARISON
comparison = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Linear": [mae_linear, mse_linear, rmse_linear, r2_linear],
    "Polynomial": [mae_poly, mse_poly, rmse_poly, r2_poly]
})

print("\nMODEL COMPARISON")
print(comparison)


# 11. LINEAR REGRESSION PLOT
plt.scatter(X, y)
plt.plot(X, linear_model.predict(X))
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()


# 12. POLYNOMIAL REGRESSION PLOT
X_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

X_grid_poly = poly.transform(X_grid)

plt.scatter(X, y)
plt.plot(X_grid, poly_model.predict(X_grid_poly))
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.show()


# 13. ACTUAL VS PREDICTED
plt.scatter(y_test, y_pred_linear)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted - Linear")
plt.show()

plt.scatter(y_test, y_pred_poly)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted - Polynomial")
plt.show()
