import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ── Load model, scaler and encoders ──────────────────────────────────────────
model     = joblib.load(os.path.join(MODEL_DIR, 'profit_model.pkl'))
scaler    = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le_crop   = joblib.load(os.path.join(MODEL_DIR, 'le_crop.pkl'))
le_season = joblib.load(os.path.join(MODEL_DIR, 'le_season.pkl'))

# ── Load real reference table ─────────────────────────────────────────────────
ref_df = pd.read_csv(os.path.join(DATA_DIR, 'reference_table.csv'))
ref_df['crop']   = ref_df['crop'].str.strip().str.title()
ref_df['state']  = ref_df['state'].str.strip().str.title()
ref_df['season'] = ref_df['season'].str.strip().str.title()

# Alias for app.py compatibility
market_df = ref_df

# All states available in real data
ALL_STATES = sorted(ref_df['state'].unique().tolist())

FEATURES = ['crop_enc', 'season_enc', 'area',
            'cost_per_hectare', 'price_per_quintal', 'yield_per_hectare']


def get_value(crop, season, state, col):
    """
    Get value for crop+season+state.
    Falls back to national average if state not found.
    """
    row = ref_df[
        (ref_df['crop']   == crop)  &
        (ref_df['season'] == season) &
        (ref_df['state']  == state)
    ]
    if not row.empty:
        return row[col].values[0], state

    row = ref_df[
        (ref_df['crop']   == crop) &
        (ref_df['season'] == season)
    ]
    if not row.empty:
        return row[col].mean(), 'National Average'

    return None, None


def predict_profit(crop: str, season: str, area: float,
                   budget: float, state: str = 'Maharashtra'):
    crop   = crop.strip().title()
    season = season.strip().title()
    state  = state.strip().title()

    # Validate
    if crop not in le_crop.classes_:
        return {"error": f"Crop '{crop}' not found in model."}
    if season not in le_season.classes_:
        return {"error": f"Season '{season}' not recognised."}

    # Get real values from reference table
    price, price_src = get_value(crop, season, state, 'modal_price_per_quintal')
    if price is None:
        return {"error": f"No data found for {crop} in {season} season."}

    yield_val, yield_src = get_value(crop, season, state, 'yield_per_hectare')
    if yield_val is None:
        return {"error": f"No yield data for {crop} in {season} season."}

    ref_cost, _ = get_value(crop, season, state, 'cost_per_hectare')

    # Cost per hectare from farmer budget
    cost_per_hectare = budget / area

    # Encode
    crop_enc   = le_crop.transform([crop])[0]
    season_enc = le_season.transform([season])[0]

    # Build feature row and scale it
    features = pd.DataFrame(
        [[crop_enc, season_enc, area,
          cost_per_hectare, price, yield_val]],
        columns=FEATURES
    )
    features_scaled = scaler.transform(features)

    # Predict
    predicted_profit  = model.predict(features_scaled)[0]
    estimated_revenue = yield_val * 10 * price * area
    total_cost        = cost_per_hectare * area

    return {
        "crop":                 crop,
        "season":               season,
        "state":                state,
        "area_hectares":        area,
        "budget":               budget,
        "cost_per_hectare":     round(cost_per_hectare, 2),
        "ref_cost_per_hectare": round(ref_cost, 2) if ref_cost else "N/A",
        "price_per_quintal":    round(price, 2),
        "price_source":         price_src,
        "yield_per_hectare":    round(yield_val, 4),
        "yield_source":         yield_src,
        "estimated_revenue":    round(estimated_revenue, 2),
        "total_cost":           round(total_cost, 2),
        "predicted_profit":     round(predicted_profit, 2),
        "verdict":              "Profitable" if predicted_profit > 0 else "Loss Expected"
    }


def recommend_top_crops(season: str, area: float, budget: float,
                         state: str = 'Maharashtra', top_n: int = 3):
    season = season.strip().title()
    state  = state.strip().title()

    available = ref_df[ref_df['season'] == season]['crop'].unique()
    results   = []

    for crop in available:
        r = predict_profit(crop, season, area, budget, state)
        if "error" not in r:
            results.append({
                "crop":              crop,
                "predicted_profit":  r["predicted_profit"],
                "price_per_quintal": r["price_per_quintal"],
                "yield_per_hectare": r["yield_per_hectare"],
                "verdict":           r["verdict"],
            })

    if not results:
        return pd.DataFrame()

    df  = pd.DataFrame(results)
    top = df.sort_values("predicted_profit", ascending=False).head(top_n)
    return top.reset_index(drop=True)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Wheat, Rabi, Punjab ===")
    r = predict_profit("Wheat", "Rabi", 2.0, 60000, "Punjab")
    for k, v in r.items():
        print(f"  {k:30} : {v}")

    print("\n=== Top 3 Kharif crops, Maharashtra ===")
    t = recommend_top_crops("Kharif", 2.0, 70000, "Maharashtra")
    print(t.to_string(index=False))
