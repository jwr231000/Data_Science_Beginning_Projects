import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
csv = r"C:\Program Files (x86)\Projects\hour.csv"
df = pd.read_csv(csv, sep=',')

# Plot the data
plt.scatter(df["casual"], df["cnt"])
plt.show()
plt.scatter(df["registered"], df["cnt"])
plt.show()

# Include more relevant features
X = df.drop(["instant", "dteday", "cnt", "casual", "registered"], axis=1)
y = df["cnt"]

# Define the preprocessing for numerical and categorical data
numerical_features = ["temp", "atemp", "hum", "windspeed"]
categorical_features = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]

# Scaling numerical data and one-hot encoding categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Build a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(random_state=42))])

# Define the parameter grid for GridSearchCV
param_grid = {
    'model__n_estimators': [2, 4, 6],
    'model__max_depth': [None, 1, 2, 3],
    'model__min_samples_split': [1, 2, 13],
    'model__min_samples_leaf': [1, 2, 3],
    'model__max_features': ['auto', 'sqrt', 'log2']
}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Cross-Validated R^2 Score:", best_score)

# Evaluate on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Set R^2 Score:", r2)
print("Test Set Mean Squared Error:", mse)