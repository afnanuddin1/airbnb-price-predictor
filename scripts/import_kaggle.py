import sqlite3
import pandas as pd

# Path to Kaggle dataset
csv_file = "data/AB_NYC_2019.csv"

# Load CSV
df = pd.read_csv(csv_file)

# Keep only useful columns
df = df[[
    "id", "neighbourhood", "room_type",
    "minimum_nights", "number_of_reviews",
    "availability_365", "price",
    "latitude", "longitude"
]]


# Connect to SQLite
conn = sqlite3.connect("db/airbnb.db")

# Write table "listings"
df.to_sql("listings", conn, if_exists="replace", index=False)

# Create predictions table if not exists
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id INTEGER,
    predicted_price REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("✅ Imported Kaggle CSV into airbnb.db (listings table created)")
