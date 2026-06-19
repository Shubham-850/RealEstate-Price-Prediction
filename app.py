
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "india_housing_prices.csv"
PRICE_MODEL_PATH = BASE_DIR / "models" / "price_model.pkl"
INVEST_MODEL_PATH = BASE_DIR / "models" / "investment_model.pkl"
META_PATH = BASE_DIR / "models" / "house_price_metadata.json"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    price_model = joblib.load(PRICE_MODEL_PATH) if PRICE_MODEL_PATH.exists() else None
    invest_model = joblib.load(INVEST_MODEL_PATH) if INVEST_MODEL_PATH.exists() else None
    return price_model, invest_model

price_model, invest_model = load_models()

# =========================
# LOAD METADATA
# =========================
if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {
        "numeric_features": [
            "BHK",
            "Size_in_SqFt",
            "Year_Built",
            "Floor_No",
            "Total_Floors",
            "Nearby_Schools",
            "Nearby_Hospitals",
            "Age_of_Property_Calc",
            "Size_per_BHK",
            "Floor_Ratio",
            "Amenity_Count",
            "Amenity_Garden",
            "Amenity_Pool",
            "Amenity_Clubhouse",
            "Amenity_Gym",
            "Amenity_Playground",
            "Is_New",
            "Is_Large",
        ],
        "categorical_features": [
            "State",
            "City",
            "Locality",
            "Property_Type",
            "Furnished_Status",
            "Public_Transport_Accessibility",
            "Parking_Space",
            "Security",
            "Facing",
            "Owner_Type",
            "Availability_Status",
        ],
        "top_amenities": ["Garden", "Pool", "Clubhouse", "Gym", "Playground"],
    }

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
top_amenities = metadata["top_amenities"]

# =========================
# HELPERS
# =========================
def build_features(input_dict: dict) -> pd.DataFrame:
    row = input_dict.copy()

    selected_amenities = row.pop("selected_amenities", [])

    row["Age_of_Property_Calc"] = max(0, 2026 - int(row["Year_Built"]))
    row["Size_per_BHK"] = float(row["Size_in_SqFt"]) / max(1, int(row["BHK"]))
    row["Floor_Ratio"] = float(row["Floor_No"]) / max(1, int(row["Total_Floors"]))
    row["Is_New"] = int(row["Age_of_Property_Calc"] <= 5)
    row["Is_Large"] = int(float(row["Size_in_SqFt"]) >= 1200)

    amenity_text = ", ".join(selected_amenities)
    row["Amenity_Count"] = 0 if not amenity_text.strip() else len([a.strip() for a in amenity_text.split(",") if a.strip()])

    for amenity in top_amenities:
        row[f"Amenity_{amenity}"] = int(amenity in selected_amenities)

    for col in numeric_features + categorical_features:
        if col not in row:
            row[col] = 0 if col in numeric_features else ""

    return pd.DataFrame([row])[numeric_features + categorical_features]

def neighborhood_score(schools, hospitals, transport, parking, security, amenity_count, age):
    transport_map = {"Low": 20, "Medium": 50, "High": 80}
    yesno_map = {"No": 0, "Yes": 10}

    score = (
        min(20, schools * 2) +
        min(20, hospitals * 2) +
        transport_map.get(transport, 0) +
        yesno_map.get(parking, 0) +
        yesno_map.get(security, 0) +
        min(15, amenity_count * 3) +
        max(0, 10 - min(age // 5, 10))
    )
    return round(min(score, 100), 1)

def estimate_growth_rate(transport, security, parking, amenity_count, age, neigh_score):
    rate = 0.045

    if neigh_score >= 80:
        rate += 0.03
    elif neigh_score >= 65:
        rate += 0.02
    elif neigh_score >= 50:
        rate += 0.01

    transport_bonus = {"Low": 0.00, "Medium": 0.01, "High": 0.02}
    yesno_bonus = {"No": 0.0, "Yes": 0.01}

    rate += transport_bonus.get(transport, 0.0)
    rate += yesno_bonus.get(security, 0.0)
    rate += yesno_bonus.get(parking, 0.0)

    if amenity_count >= 4:
        rate += 0.02
    elif amenity_count >= 2:
        rate += 0.01

    if age <= 5:
        rate += 0.015
    elif age >= 20:
        rate -= 0.01

    return float(np.clip(rate, 0.03, 0.15))

def make_similar_properties(user_row: pd.Series, source_df: pd.DataFrame, top_n: int = 5):
    temp = source_df.copy()
    temp = temp[temp["City"] == user_row["City"]]
    temp = temp[temp["Property_Type"] == user_row["Property_Type"]]
    temp = temp[temp["BHK"].between(max(1, user_row["BHK"] - 1), user_row["BHK"] + 1)]

    size_low = user_row["Size_in_SqFt"] * 0.8
    size_high = user_row["Size_in_SqFt"] * 1.2
    temp = temp[temp["Size_in_SqFt"].between(size_low, size_high)]

    if temp.empty:
        return temp

    temp = temp.copy()
    temp["size_diff"] = (temp["Size_in_SqFt"] - user_row["Size_in_SqFt"]).abs()
    temp["bhk_diff"] = (temp["BHK"] - user_row["BHK"]).abs()
    temp["score"] = temp["size_diff"] + temp["bhk_diff"] * 100

    cols = [
        "State", "City", "Locality", "Property_Type", "BHK",
        "Size_in_SqFt", "Price_in_Lakhs", "Furnished_Status",
        "Facing", "Owner_Type"
    ]
    return temp.sort_values("score").head(top_n)[cols]

# =========================
# TITLE
# =========================
st.title("🏠 House Price Predictor")
st.write("Enter property details to predict the current price, 5-year future price, and investment quality.")

if price_model is None or invest_model is None:
    st.error("Model files are missing. First run the training notebook and save the models inside the /models folder.")
    st.stop()

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Enter Property Details")

state_options = sorted(df["State"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("State", state_options)

city_options = sorted(df.loc[df["State"] == selected_state, "City"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("City", city_options)

locality_options = sorted(
    df.loc[(df["State"] == selected_state) & (df["City"] == selected_city), "Locality"]
    .dropna()
    .unique()
    .tolist()
)
selected_locality = st.sidebar.selectbox("Locality", locality_options)

property_type = st.sidebar.selectbox("Property Type", sorted(df["Property_Type"].dropna().unique().tolist()))
bhk = st.sidebar.slider("BHK", int(df["BHK"].min()), int(df["BHK"].max()), 2)
size_sqft = st.sidebar.slider("Size in SqFt", int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max()), 1200)
year_built = st.sidebar.slider("Year Built", int(df["Year_Built"].min()), int(df["Year_Built"].max()), 2015)
floor_no = st.sidebar.slider("Floor No", int(df["Floor_No"].min()), int(df["Floor_No"].max()), 5)
total_floors = st.sidebar.slider("Total Floors", int(df["Total_Floors"].min()), int(df["Total_Floors"].max()), 10)
furnished = st.sidebar.selectbox("Furnished Status", sorted(df["Furnished_Status"].dropna().unique().tolist()))
near_schools = st.sidebar.slider("Nearby Schools", int(df["Nearby_Schools"].min()), int(df["Nearby_Schools"].max()), 5)
near_hospitals = st.sidebar.slider("Nearby Hospitals", int(df["Nearby_Hospitals"].min()), int(df["Nearby_Hospitals"].max()), 3)
transport = st.sidebar.selectbox("Public Transport Accessibility", sorted(df["Public_Transport_Accessibility"].dropna().unique().tolist()))
parking = st.sidebar.selectbox("Parking Space", sorted(df["Parking_Space"].dropna().unique().tolist()))
security = st.sidebar.selectbox("Security", sorted(df["Security"].dropna().unique().tolist()))
facing = st.sidebar.selectbox("Facing", sorted(df["Facing"].dropna().unique().tolist()))
owner_type = st.sidebar.selectbox("Owner Type", sorted(df["Owner_Type"].dropna().unique().tolist()))
availability = st.sidebar.selectbox("Availability Status", sorted(df["Availability_Status"].dropna().unique().tolist()))
selected_amenities = st.sidebar.multiselect("Amenities", top_amenities, default=top_amenities[:2])

predict_btn = st.sidebar.button("Predict")

# =========================
# MAIN
# =========================
age = max(0, 2026 - year_built)
amenity_count = len(selected_amenities)
neigh_score = neighborhood_score(
    near_schools, near_hospitals, transport, parking, security, amenity_count, age
)

city_avg = df[(df["City"] == selected_city) & (df["Property_Type"] == property_type)]["Price_in_Lakhs"].mean()
if np.isnan(city_avg):
    city_avg = df[df["City"] == selected_city]["Price_in_Lakhs"].mean()

user_input = {
    "State": selected_state,
    "City": selected_city,
    "Locality": selected_locality,
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": size_sqft,
    "Year_Built": year_built,
    "Floor_No": floor_no,
    "Total_Floors": total_floors,
    "Nearby_Schools": near_schools,
    "Nearby_Hospitals": near_hospitals,
    "Furnished_Status": furnished,
    "Public_Transport_Accessibility": transport,
    "Parking_Space": parking,
    "Security": security,
    "Facing": facing,
    "Owner_Type": owner_type,
    "Availability_Status": availability,
    "selected_amenities": selected_amenities,
}

if predict_btn:
    input_df = build_features(user_input)

    current_price = float(price_model.predict(input_df)[0])
    invest_prob = float(invest_model.predict_proba(input_df)[0, 1])

    growth_rate = estimate_growth_rate(
        transport=transport,
        security=security,
        parking=parking,
        amenity_count=amenity_count,
        age=age,
        neigh_score=neigh_score,
    )
    future_price = current_price * ((1 + growth_rate) ** 5)
    expected_growth = ((future_price - current_price) / current_price) * 100

    market_diff = current_price - float(city_avg)
    market_status = "Below market average" if market_diff < 0 else "Above market average"

    if invest_prob >= 0.60 and expected_growth >= 25:
        investment_status = "Good Investment"
    elif invest_prob >= 0.50 and expected_growth >= 15:
        investment_status = "Moderate Investment"
    else:
        investment_status = "Not a Strong Investment"

    # Header metrics
    st.subheader("Prediction Result")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Current Price", f"₹ {current_price:,.2f} Lakhs")
    c2.metric("5-Year Price", f"₹ {future_price:,.2f} Lakhs")
    c3.metric("Expected Growth", f"{expected_growth:.1f}%")
    c4.metric("Investment Probability", f"{invest_prob * 100:.1f}%")

    st.info(f"Market comparison: {market_status} (vs city average {city_avg:,.2f} lakhs)")

    # Investment verdict
    st.subheader("Investment Analysis")
    if investment_status == "Good Investment":
        st.success(f"✅ {investment_status}")
    elif investment_status == "Moderate Investment":
        st.info(f"ℹ️ {investment_status}")
    else:
        st.warning(f"⚠️ {investment_status}")

    # Dashboard
    st.subheader("Performance Dashboard")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Neighborhood Score", f"{neigh_score}/100")
    d2.metric("Amenities Selected", f"{amenity_count}")
    d3.metric("Feature Count", f"{len(numeric_features) + len(categorical_features)}")
    d4.metric("Projected Growth Rate", f"{growth_rate * 100:.1f}% / year")

    # Chart
    price_df = pd.DataFrame(
        {"Price": [current_price, future_price]},
        index=["Current", "After 5 Years"]
    )
    st.bar_chart(price_df)

    # Similar properties
    st.subheader("Similar Properties Nearby")
    similar = make_similar_properties(pd.Series(user_input), df, top_n=5)
    if similar.empty:
        st.warning("No close matches found with the current filters.")
    else:
        st.dataframe(similar, use_container_width=True)

    # Log
    log_row = {
        "Date Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "State": selected_state,
        "City": selected_city,
        "Locality": selected_locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Year_Built": year_built,
        "Nearby_Schools": near_schools,
        "Nearby_Hospitals": near_hospitals,
        "Transport": transport,
        "Parking": parking,
        "Security": security,
        "Amenities": ", ".join(selected_amenities),
        "Current Price (Lakhs)": round(current_price, 2),
        "5 Year Price (Lakhs)": round(future_price, 2),
        "Expected Growth %": round(expected_growth, 2),
        "Investment Probability": round(invest_prob, 3),
        "Investment Status": investment_status,
        "Neighborhood Score": neigh_score,
    }

    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    st.session_state["logs"].append(log_row)

    st.subheader("Prediction Log")
    log_df = pd.DataFrame(st.session_state["logs"])
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction Log",
        data=csv,
        file_name="house_price_prediction_log.csv",
        mime="text/csv",
    )

else:
    st.subheader("Current Input Summary")
    summary = pd.DataFrame([{
        "State": selected_state,
        "City": selected_city,
        "Locality": selected_locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Year_Built": year_built,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Nearby_Schools": near_schools,
        "Nearby_Hospitals": near_hospitals,
        "Furnished_Status": furnished,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Amenities": ", ".join(selected_amenities),
        "Age_of_Property": age,
        "Neighborhood_Score": neigh_score,
    }])
    st.dataframe(summary, use_container_width=True)

    st.caption("Click 'Predict' in the sidebar to generate the outputs.")
    st.caption("Future price is an estimate based on feature-driven annual growth, because the dataset does not contain time-series prices.")
