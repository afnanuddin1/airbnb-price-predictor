import pandas as pd

df = pd.read_csv("data/processed.csv")


print("Data Loaded:", df.shape)
print(df.head())

from sklearn.model_selection import train_test_split


features = ["neighbourhood", "room_type", "minimum_nights",
            "number_of_reviews", "availability_365"]

target = "price"

X = df[features]
y = df[target]

print("Features and Target selected")
print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical = ["neighbourhood", "room_type"]
numerical = ["minimum_nights", "number_of_reviews", "availability_365"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numerical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test split done")
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
print("Model trained")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Model RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "models/price_model.pkl")
print("Model saved to models/price_model.pkl")
