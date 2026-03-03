import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("models/price_model.pkl")

# Load processed dataset to get options
df = pd.read_csv("data/processed.csv")
unique_neighbourhoods = sorted(df["neighbourhood"].unique().tolist())
unique_room_types = sorted(df["room_type"].unique().tolist())

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡")
st.title("🏡 Airbnb Price Predictor")
st.write("Enter listing details of an Airbnb in NYC and get an estimated nightly price.")

# --- User Inputs ---
neighbourhood = st.selectbox("Neighbourhood", unique_neighbourhoods)
room_type = st.selectbox("Room Type", unique_room_types)
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=3)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=20)
availability_365 = st.slider("Availability (days per year)", 0, 365, 150)

# --- Prediction Button ---
if st.button("Predict Price"):
    new_listing = pd.DataFrame([{
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "availability_365": availability_365
    }])

    predicted_price = model.predict(new_listing)[0]
    st.success(f"💰 Estimated Nightly Price: **${predicted_price:.2f}**")

    # --- Show price comparison ---
    st.subheader("📊 Price Comparison")

    # Calculate averages
    neighbourhood_avg = df[df["neighbourhood"] == neighbourhood]["price"].mean()
    roomtype_avg = df[df["room_type"] == room_type]["price"].mean()
    city_avg = df["price"].mean()

    comparison_df = pd.DataFrame({
        "Category": ["Predicted Price", f"{neighbourhood.title()} Avg", f"{room_type.title()} Avg", "City Avg"],
        "Price": [predicted_price, neighbourhood_avg, roomtype_avg, city_avg]
    })
    # Show bar chart
    st.bar_chart(comparison_df.set_index("Category"))
    

    # --- Show feature importance ---
    st.subheader("🔍 Feature Importance")

    # Get feature names after preprocessing
    ohe = model.named_steps["preprocessor"].transformers_[0][1]  # OneHotEncoder
    ohe_features = ohe.get_feature_names_out(["neighbourhood", "room_type"])
    all_features = list(ohe_features) + ["minimum_nights", "number_of_reviews", "availability_365"]

    # Get importance values from RandomForest
    importances = model.named_steps["regressor"].feature_importances_

    # Create DataFrame
    fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(10)  # top 10

    st.bar_chart(fi_df.set_index("Feature"))
    # --- Show map of listings in this neighbourhood ---
    st.subheader(f"🗺️ Map of Listings in {neighbourhood.title()}")

    # Filter dataset for selected neighbourhood
    map_df = df[df["neighbourhood"] == neighbourhood].copy()

    # Only keep rows with latitude/longitude (the original Kaggle dataset has them)
    if "latitude" in map_df.columns and "longitude" in map_df.columns:
        map_df = map_df[["latitude", "longitude", "price"]]

        st.map(map_df)

        st.caption("Each dot represents a listing. Larger datasets may render slowly.")
    else:
        st.warning("⚠️ Latitude/Longitude columns not found in your processed dataset.")
