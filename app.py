import streamlit as st
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from predict import predict_profit, recommend_top_crops, ref_df, ALL_STATES

st.set_page_config(
    page_title="Yield2Profit",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Nunito:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: #f7f5ef !important;
    color: #2f3b2a !important;
}

[data-testid="stHeader"]  { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer         { visibility: hidden !important; }
.block-container          { padding: 1rem 1.2rem 3rem !important; max-width: 700px !important; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 1.8rem 1rem 1.5rem;
    border-bottom: 3px solid #c9a14a;
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #3f5b2f;
    margin: 0;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #6b7d5e;
    margin-top: 0.3rem;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Lora', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #3f5b2f;
    margin: 1.6rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #dfe6d3;
}

/* ── Input panel ── */
.input-panel {
    background: #ffffff;
    border: 1px solid #e0ddd0;
    border-radius: 12px;
    padding: 1.4rem 1.4rem 0.6rem;
    margin-bottom: 1rem;
}

/* ── Widget labels ── */
label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #3f5b2f !important;
}

/* ── Select / number boxes ── */
[data-testid="stSelectbox"] > div > div {
    background: #fbfaf6 !important;
    border: 1px solid #cfcabb !important;
    border-radius: 8px !important;
    color: #2f3b2a !important;
    font-size: 1rem !important;
}
[data-testid="stNumberInput"] input {
    background: #fbfaf6 !important;
    border: 1px solid #cfcabb !important;
    border-radius: 8px !important;
    color: #2f3b2a !important;
    font-size: 1rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #4a6b3a !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.4rem !important;
    width: 100% !important;
    transition: background-color 0.2s !important;
}
.stButton > button:hover {
    background-color: #3a5a2b !important;
}

/* ── Result box ── */
.result-box {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    text-align: center;
}
.result-profit {
    background-color: #eaf3e3;
    border: 1px solid #b9d4a4;
}
.result-loss {
    background-color: #fbeceb;
    border: 1px solid #e3b3af;
}
.result-tag {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.result-profit .result-tag { color: #4a6b3a; }
.result-loss   .result-tag { color: #b3413b; }
.result-amount {
    font-family: 'Lora', serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0.2rem 0;
}
.result-profit .result-amount { color: #3f5b2f; }
.result-loss   .result-amount { color: #b3413b; }
.result-sub {
    font-size: 0.9rem;
    color: #6b7d5e;
}

/* ── Info chips ── */
.chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 0.8rem 0;
    justify-content: center;
}
.chip {
    background: #f0ede2;
    border: 1px solid #ddd8c8;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 0.82rem;
    color: #4a6b3a;
    font-weight: 600;
}

/* ── Stat table ── */
.stat-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff;
    border: 1px solid #e0ddd0;
    border-radius: 10px;
    overflow: hidden;
    margin: 0.8rem 0;
}
.stat-table td {
    padding: 0.7rem 1rem;
    font-size: 0.95rem;
    border-bottom: 1px solid #ece9dd;
}
.stat-table tr:last-child td { border-bottom: none; }
.stat-table .label { color: #6b7d5e; font-weight: 500; }
.stat-table .value { text-align: right; font-weight: 700; color: #3f5b2f; }

/* ── Tip box ── */
.tip-box {
    background: #f3f7ee;
    border-left: 4px solid #9bbf85;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    font-size: 0.9rem;
    color: #4a5d3d;
    margin: 0.8rem 0;
}
.tip-warn {
    background: #fdf6ec;
    border-left: 4px solid #d9a85c;
    color: #7a5a25;
}

/* ── Rank rows ── */
.rank-row {
    background: #ffffff;
    border: 1px solid #e0ddd0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.rank-medal { font-size: 1.6rem; min-width: 36px; text-align: center; }
.rank-body { flex: 1; }
.rank-name {
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #2f3b2a;
}
.rank-meta { font-size: 0.82rem; color: #6b7d5e; }
.rank-profit {
    font-family: 'Lora', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #3f5b2f;
    white-space: nowrap;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.8rem;
    color: #9aa68c;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e0ddd0;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🌾 Yield2Profit</div>
    <div class="app-subtitle">Know your crop profit before you plant</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT SECTION ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">Enter Your Crop Details</div>', unsafe_allow_html=True)

st.markdown('<div class="input-panel">', unsafe_allow_html=True)

season = st.selectbox("Season", ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Autumn', 'Winter'])

available_crops = sorted(
    ref_df[ref_df['season'].str.title() == season]['crop'].str.title().unique()
)
if not available_crops:
    available_crops = sorted(ref_df['crop'].str.title().unique())

crop  = st.selectbox("Crop Type", available_crops)
state = st.selectbox("State", ALL_STATES)
area  = st.number_input("Land Area (Hectares)", min_value=0.1, max_value=50.0,
                         value=1.0, step=0.1, help="1 hectare = 2.47 acres")
budget = st.number_input("Total Budget (₹)", min_value=1000, max_value=10000000,
                          value=50000, step=1000)

st.markdown('</div>', unsafe_allow_html=True)

acres = round(area * 2.47, 2)
st.markdown(f"""
<div class="chip-row">
    <span class="chip">📐 {acres} acres</span>
    <span class="chip">📍 {state}</span>
    <span class="chip">💰 ₹{int(budget/area):,}/hectare</span>
</div>
""", unsafe_allow_html=True)

# ── BUTTONS ───────────────────────────────────────────────────────────────────
b1, b2 = st.columns(2)
with b1:
    predict_btn = st.button("Predict My Profit")
with b2:
    rec_btn = st.button("Show Best Crops")

# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analysing market data..."):
        result = predict_profit(crop, season, area, budget, state)

    if "error" in result:
        st.error(result["error"])
    else:
        profit    = result['predicted_profit']
        is_profit = profit > 0
        cls  = "result-profit" if is_profit else "result-loss"
        tag  = "Profitable Crop" if is_profit else "Loss Expected"
        sign = "+" if is_profit else "−"

        st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box {cls}">
            <div class="result-tag">{tag}</div>
            <div class="result-amount">{sign} ₹{abs(profit):,.0f}</div>
            <div class="result-sub">{crop} · {season} · {area} ha · 📍 {state}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="chip-row">
            <span class="chip">Price source: {result['price_source']}</span>
            <span class="chip">Yield source: {result['yield_source']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Financial Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="stat-table">
            <tr><td class="label">Estimated Revenue</td><td class="value">₹{result['estimated_revenue']:,.0f}</td></tr>
            <tr><td class="label">Total Cost</td><td class="value">₹{result['total_cost']:,.0f}</td></tr>
            <tr><td class="label">Net Profit</td><td class="value">{sign}₹{abs(profit):,.0f}</td></tr>
            <tr><td class="label">Market Price</td><td class="value">₹{result['price_per_quintal']:,.0f} / quintal</td></tr>
            <tr><td class="label">Average Yield</td><td class="value">{result['yield_per_hectare']} tons / hectare</td></tr>
            <tr><td class="label">Cost per Hectare</td><td class="value">₹{result['cost_per_hectare']:,.0f}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        if is_profit:
            roi = round((profit / result['total_cost']) * 100, 1)
            st.markdown(f"""
            <div class="tip-box">
                Your estimated return on investment (ROI) is <b>{roi}%</b>.
                This crop looks like a good choice for the selected season and state.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tip-box tip-warn">
                Your budget is high compared to expected revenue for this crop.
                Try reducing your budget, choosing a smaller area, or click
                <b>Show Best Crops</b> to see better alternatives.
            </div>
            """, unsafe_allow_html=True)

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
if rec_btn:
    with st.spinner("Finding best crops for this season and state..."):
        top3 = recommend_top_crops(season, area, budget, state, top_n=3)

    st.markdown(
        f'<div class="section-heading">Best Crops — {season} · {state}</div>',
        unsafe_allow_html=True
    )

    if top3.empty:
        st.warning("No crop data found for this season and state combination.")
    else:
        medals = ["🥇", "🥈", "🥉"]
        for i, row in top3.iterrows():
            st.markdown(f"""
            <div class="rank-row">
                <div class="rank-medal">{medals[i]}</div>
                <div class="rank-body">
                    <div class="rank-name">{row['crop']}</div>
                    <div class="rank-meta">
                        ₹{row['price_per_quintal']:,.0f}/quintal · {row['yield_per_hectare']} T/ha
                    </div>
                </div>
                <div class="rank-profit">+₹{row['predicted_profit']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="tip-box">
            Based on real mandi prices for <b>{state}</b>, your land area of
            <b>{area} ha</b>, and budget of <b>₹{budget:,}</b>.
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Yield2Profit · Empowering Indian Farmers with AI<br>
    Alisha Kumari · PES1PG25CA375
</div>
""", unsafe_allow_html=True)