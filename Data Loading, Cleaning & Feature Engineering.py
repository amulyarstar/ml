# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 2. LOAD DATASET
data = pd.read_csv("winequality-red.csv")

print("First 5 Rows:")
print(data.head())

print("\nShape:", data.shape)
print("\nColumns:")
print(data.columns)

print("\nDataset Information:")
data.info()

print("\nStatistical Summary:")
print(data.describe())


# 3. CHECK MISSING VALUES
print("\nMissing Values:")
print(data.isnull().sum())

# Separate features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Mean Imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

print("\nMissing Values After Imputation:")
print(pd.DataFrame(X).isnull().sum())


# 4. REMOVE DUPLICATES
print("\nDuplicate Rows:", data.duplicated().sum())

data = data.drop_duplicates()

print("Shape After Removing Duplicates:", data.shape)


# 5. DETECT OUTLIERS USING BOXPLOT
plt.figure(figsize=(14, 8))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title("Outlier Detection")
plt.show()


# 6. REMOVE OUTLIERS USING IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data_clean = data[
    ~((data < (Q1 - 1.5 * IQR)) |
      (data > (Q3 + 1.5 * IQR))).any(axis=1)
]

print("\nOriginal Shape:", data.shape)
print("After Removing Outliers:", data_clean.shape)


# 7. STANDARDIZATION
X = data_clean.drop("quality", axis=1)
y = data_clean["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStandardized Features:")
print(X_scaled[:5])


# 8. VISUALIZATION
plt.figure(figsize=(8, 5))
sns.histplot(data_clean["quality"], kde=True)
plt.title("Quality Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(data_clean.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
