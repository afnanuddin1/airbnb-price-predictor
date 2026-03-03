import sqlite3
import pandas as pd
import joblib

model = joblib.load("models/price_model.pkl")
print("Model loaded")

# Example input data
new_listings = pd.DataFrame([
    {
        "neighbourhood": "brooklyn",
        "room_type": "Entire home/apt",
        "minimum_nights": 3,
        "number_of_reviews": 20,
        "availability_365": 150
    }])

# Predict prices
predicted_prices = model.predict(new_listings)[0]
print(f"Predicted Price: ${predicted_prices:.2f}")

# Save prediction to DB
conn = sqlite3.connect("db/airbnb.db")
cursor = conn.cursor()

cursor.execute("""
INSERT INTO predictions (listing_id, predicted_price)
VALUES (?, ?)
""", (99999, predicted_prices)) # 99999 is a dummy listing_id

conn.commit()
conn.close()

print("Prediction saved to DB")