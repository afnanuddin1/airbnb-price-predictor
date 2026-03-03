# Airbnb NYC Price Predictor 🏡

A machine learning web app that predicts nightly Airbnb prices in New York City based on listing details. Built with Streamlit and a Random Forest model trained on the [NYC Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) dataset.

## Features

- Predict nightly price from neighbourhood, room type, and availability details
- Compare your predicted price against neighbourhood, room type, and city averages
- Visualize top feature importances from the trained model
- Map of listings in the selected neighbourhood

## Project Structure

```
airbnb-price-predict/
├── app/
│   └── streamlit_app.py      # Streamlit web app
├── scripts/
│   ├── import_kaggle.py      # Load Kaggle CSV into SQLite
│   ├── preprocess.py         # Clean data and export processed CSV
│   ├── train_model.py        # Train and save the RandomForest model
│   └── predict.py            # Standalone prediction script
├── data/
│   └── AB_NYC_2019.csv       # Raw Kaggle dataset (not tracked in git)
├── db/
│   └── airbnb.db             # SQLite database (not tracked in git)
├── models/
│   └── price_model.pkl       # Trained model (not tracked in git)
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

### 2. Download the dataset

Download [AB_NYC_2019.csv](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) from Kaggle and place it in the `data/` folder.

### 3. Import data into SQLite

```bash
python scripts/import_kaggle.py
```

### 4. Preprocess the data

```bash
python scripts/preprocess.py
```

### 5. Train the model

```bash
python scripts/train_model.py
```

### 6. Run the app

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Model

- **Algorithm:** Random Forest Regressor (100 estimators)
- **Features:** neighbourhood, room type, minimum nights, number of reviews, availability (days/year)
- **Target:** nightly price (USD)
- **Preprocessing:** One-hot encoding for categorical features, prices filtered to $1–$1000

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- SQLite
- joblib
