import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("used_car_dataset.csv")


# Clean kmDriven
# Example: "98,000 km" -> 98000
df["kmDriven"] = (
    df["kmDriven"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(" km", "", regex=False)
)

df["kmDriven"] = pd.to_numeric(df["kmDriven"], errors="coerce")


# Clean AskPrice
# Example: "₹ 1,95,000" -> 195000
df["AskPrice"] = (
    df["AskPrice"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["AskPrice"] = pd.to_numeric(df["AskPrice"], errors="coerce")


# Remove rows where target price is missing
df = df.dropna(subset=["AskPrice"])


# Features
# Age is intentionally removed
features = [
    "Brand",
    "model",
    "Year",
    "kmDriven",
    "Transmission",
    "Owner",
    "FuelType"
]

X = df[features]
y = df["AskPrice"]


# Numerical columns
numerical_columns = [
    "Year",
    "kmDriven"
]


# Categorical columns
categorical_columns = [
    "Brand",
    "model",
    "Transmission",
    "Owner",
    "FuelType"
]


# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train
pipeline.fit(X_train, y_train)


# Evaluate
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Complete!")
print("MAE:", mae)
print("R²:", r2)


# Save model
joblib.dump(pipeline, "used_car_price_model.pkl")

print("Model saved as used_car_price_model.pkl")