import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Dubai Real Estate Price Predictor",
    page_icon="🏙️",
    layout="wide",
)

st.sidebar.title("Dubai Property Price App")

st.sidebar.markdown(
    """
**How it works**

- If you **do nothing**, the app trains on a built‑in UAE 2024 sample dataset.  
- If you **upload a CSV** with a similar schema, it trains on your data instead.

**Expected columns (or similar names)**:
- `price`
- `type` (must include *Residential for Sale*)
- `sizeMin` (e.g. `"1,200 sqft"`)
- `bathrooms`
- `bedrooms`
- `furnishing`
"""
)

# ----------------- DATA LOADING -----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload real estate CSV (optional)", type=["csv"]
)

@st.cache_data
def load_data(file_or_path, nrows=10000):
    """Load and clean raw listings data (works for path or uploaded file)."""
    df = pd.read_csv(file_or_path, nrows=nrows)

    # --- price cleaning ---
    def clean_price(price):
        try:
            price = str(price).replace(",", "").replace("AED", "").strip()
            return float(re.sub(r"[^0-9.]", "", price))
        except Exception:
            return np.nan

    df["price"] = df["price"].apply(clean_price)
    df = df.dropna(subset=["price"])

    # only residential for sale listings
    if "type" in df.columns:
        df = df[df["type"] == "Residential for Sale"]

    # size in sqft from sizeMin like "1,200 sqft"
    df["size_sqft"] = (
        df["sizeMin"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)", expand=False)
        .astype(float)
    )

    # numeric features: bathrooms, bedrooms, size_sqft
    for col in ["bathrooms", "bedrooms", "size_sqft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    # furnishing: normalize to Yes / No
    df["furnishing"] = (
        df.get("furnishing", pd.Series(["No"] * len(df)))
        .fillna("No")
        .astype(str)
        .str.upper()
        .map({"YES": "Yes", "NO": "No"})
        .fillna("No")
    )

    return df

# Decide data source
if uploaded_file is not None:
    raw_source = uploaded_file
    source_label = "Uploaded dataset"
else:
    raw_source = "uae_real_estate_2024.csv"   # file shipped with the app
    source_label = "Built‑in UAE 2024 sample"

with st.spinner(f"Loading and processing: {source_label}…"):
    data = load_data(raw_source)

st.success(f"{source_label} loaded. {len(data):,} records after cleaning.")

# ----------------- MODEL TRAINING -----------------
@st.cache_resource
def train_model(df):
    features = ["bathrooms", "bedrooms", "size_sqft", "furnishing"]
    X = df[features]
    y = df["price"]

    numeric_features = ["bathrooms", "bedrooms", "size_sqft"]
    categorical_features = ["furnishing"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=80,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X, y)
    return pipe

model = train_model(data)

# ----------------- LAYOUT -----------------
st.markdown(
    "<h2 style='text-align:center;color:#145DA0;'>🏠 Dubai Property Price Predictor</h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Enter property features to estimate the expected sale price. "
    "Model is retrained on the selected dataset (built‑in or uploaded)."
)

left, mid, right = st.columns([1, 2, 1])

with mid:
    st.subheader("Property details")

    bathrooms = st.slider("Bathrooms", 0, 10, 2)
    bedrooms = st.slider("Bedrooms", 0, 10, 2)
    size_sqft = st.number_input("Size (sqft)", 200, 20000, 1000, step=50)
    furnishing = st.radio("Furnishing", ["Yes", "No"], horizontal=True)

    input_df = pd.DataFrame(
        {
            "bathrooms": [bathrooms],
            "bedrooms": [bedrooms],
            "size_sqft": [size_sqft],
            "furnishing": [furnishing],
        }
    )

    if st.button("🎬 Predict price"):
        pred = model.predict(input_df)[0]
        st.markdown(
            f"""
            <div style='padding:16px;background:#F0F5F9;border-radius:12px;
                        text-align:center;margin-top:20px;'>
                <h3 style='color:#145DA0;'>💰 Estimated price: AED {pred:,.0f}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- PRICE DISTRIBUTION -----------------
st.markdown("---")
with st.expander("📊 View sample price distribution"):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(data["price"], bins=40, color="#145DA0", ax=ax)
    ax.set_xlabel("Price (AED)")
    ax.set_ylabel("Count")
    ax.set_title("Sample price distribution")
    st.pyplot(fig)

st.markdown(
    """
---
<div style='text-align:center;color:grey;font-size:13px;margin-top:10px;'>
Made with ❤️ in Dubai | Powered by Streamlit
</div>
""",
    unsafe_allow_html=True,
)