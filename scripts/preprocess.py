import sqlite3
import pandas as pd

conn = sqlite3.connect("db/airbnb.db")

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in DB:", cursor.fetchall())

df = pd.read_sql_query("SELECT * FROM listings", conn)

print("Data Loaded:", df.shape)


df = df[df["price"] > 0]

df=df[df["price"] < 1000]

# Fill missing values
if "neighbourhood" in df.columns:
    df["neighbourhood"] = df["neighbourhood"].fillna("").astype(str).str.lower()

if "room_type" in df.columns:
    df["room_type"] = df["room_type"].fillna("").astype(str).str.lower()


# Convert column text to lowercase
df["neighbourhood"] = df["neighbourhood"].fillna("").astype(str).str.lower()
df["room_type"] = df["room_type"].fillna("").astype(str).str.lower()

print("Data Cleaned:", df.shape)

# Keep only relevant columns
features = ["neighbourhood", "room_type", "minimum_nights",
            "number_of_reviews", "availability_365",
            "latitude", "longitude"]

target = "price"

df = df[features + [target]]

print("Final Data: ", df.shape)

# Save processed data
df.to_csv("data/processed.csv", index=False)




# Close DB
conn.close()