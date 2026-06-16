import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── LOAD REAL DATASETS ────────────────────────────────────────────────────────
print("Loading real datasets...")

mandi_df = pd.read_csv(os.path.join(DATA_DIR, 'mandi_prices_real.csv'))
yield_df = pd.read_csv(os.path.join(DATA_DIR, 'crop_yield_real.csv'))
print(f"✅ mandi_prices_real : {mandi_df.shape}")
print(f"✅ crop_yield_real   : {yield_df.shape}")

# ── CLEAN MANDI PRICES ────────────────────────────────────────────────────────
print("\nCleaning mandi prices...")

mandi_df.columns = [c.strip().lower().replace(' ', '_') for c in mandi_df.columns]
mandi_df.rename(columns={
    'commodity':   'crop',
    'modal_price': 'modal_price_per_quintal',
    'price_date':  'price_date',
}, inplace=True)

mandi_df['crop']  = mandi_df['crop'].str.strip().str.title()
mandi_df['state'] = mandi_df['state'].str.strip().str.title()

mandi_df.dropna(subset=['modal_price_per_quintal'], inplace=True)
mandi_df = mandi_df[mandi_df['modal_price_per_quintal'] > 0]

# Average modal price per crop+state across all mandis and dates
market_prices = (
    mandi_df.groupby(['crop', 'state'])['modal_price_per_quintal']
    .mean()
    .reset_index()
)
market_prices['modal_price_per_quintal'] = market_prices['modal_price_per_quintal'].round(2)
print(f"✅ Unique crop+state market price combos: {len(market_prices)}")
print(market_prices.head(5))

# ── CLEAN YIELD DATA ──────────────────────────────────────────────────────────
print("\nCleaning yield data...")

yield_df.columns = [c.strip().lower().replace(' ', '_') for c in yield_df.columns]
yield_df.rename(columns={'yield': 'yield_per_hectare'}, inplace=True)

yield_df['crop']   = yield_df['crop'].str.strip().str.title()
yield_df['state']  = yield_df['state'].str.strip().str.title()
yield_df['season'] = yield_df['season'].str.strip().str.title()

yield_df.dropna(subset=['yield_per_hectare'], inplace=True)
yield_df = yield_df[yield_df['yield_per_hectare'] > 0]

# Average yield per crop+state+season
yield_avg = (
    yield_df.groupby(['crop', 'state', 'season'])['yield_per_hectare']
    .mean()
    .reset_index()
)
yield_avg['yield_per_hectare'] = yield_avg['yield_per_hectare'].round(4)
print(f"✅ Unique crop+state+season yield combos: {len(yield_avg)}")
print(yield_avg.head(5))

# ── CROP NAME MAPPING ─────────────────────────────────────────────────────────
# Map mandi crop names → yield dataset crop names so merge works
print("\nMapping crop names...")

crop_name_map = {
    'Arhar/Tur':         'Arhar/Tur',
    'Wheat':             'Wheat',
    'Rice':              'Rice',
    'Maize':             'Maize',
    'Cotton':            'Cotton(Lint)',
    'Groundnut':         'Groundnut',
    'Soyabean':          'Soyabean',
    'Soybean':           'Soyabean',
    'Sunflower':         'Sunflower',
    'Mustard':           'Rapeseed &Mustard',
    'Rapeseed':          'Rapeseed &Mustard',
    'Potato':            'Potato',
    'Tomato':            'Tomato',
    'Onion':             'Onion',
    'Sugarcane':         'Sugarcane',
    'Jowar':             'Jowar',
    'Bajra':             'Bajra',
    'Ragi':              'Ragi',
    'Gram':              'Gram',
    'Moong(Green Gram)': 'Moong(Green Gram)',
    'Urad':              'Urad',
    'Lentil':            'Lentil',
    'Ginger':            'Ginger',
    'Turmeric':          'Turmeric',
    'Banana':            'Banana',
    'Coconut':           'Coconut',
    'Jute':              'Jute',
    'Castor Seed':       'Castor Seed',
    'Sesamum':           'Sesamum',
    'Linseed':           'Linseed',
    'Coffee':            'Coffee',
    'Tea':               'Tea',
    'Arecanut':          'Arecanut',
}

market_prices['crop_mapped'] = (
    market_prices['crop'].map(crop_name_map).fillna(market_prices['crop'])
)

# ── MERGE YIELD + MARKET PRICES ───────────────────────────────────────────────
print("\nMerging datasets...")

merged = yield_avg.merge(
    market_prices[['crop_mapped', 'state', 'modal_price_per_quintal']].rename(
        columns={'crop_mapped': 'crop'}
    ),
    on=['crop', 'state'],
    how='inner'
)
print(f"After yield + market price merge: {merged.shape}")

# ── FIXED COST OF CULTIVATION (CACP published averages ₹/hectare) ─────────────
# Source: CACP Cost of Cultivation reports — A2+FL cost concept
fixed_cost_map = {
    'Rice':              35000,
    'Wheat':             30000,
    'Maize':             22000,
    'Jowar':             15000,
    'Bajra':             13000,
    'Ragi':              16000,
    'Arhar/Tur':         18000,
    'Moong(Green Gram)': 17000,
    'Urad':              16000,
    'Gram':              16000,
    'Lentil':            15000,
    'Groundnut':         28000,
    'Soyabean':          20000,
    'Sunflower':         18000,
    'Rapeseed &Mustard': 14000,
    'Sesamum':           14000,
    'Cotton(Lint)':      38000,
    'Sugarcane':         80000,
    'Jute':              30000,
    'Onion':             50000,
    'Potato':            55000,
    'Tomato':            60000,
    'Ginger':            90000,
    'Turmeric':          70000,
    'Banana':            120000,
    'Coconut':           40000,
    'Coffee':            85000,
    'Tea':               100000,
    'Castor Seed':       18000,
    'Linseed':           14000,
    'Arecanut':          80000,
}

merged['cost_per_hectare'] = merged['crop'].map(fixed_cost_map)

# Fallback: use 55% of revenue as cost for any crop not in map
mask = merged['cost_per_hectare'].isna()
merged.loc[mask, 'cost_per_hectare'] = (
    merged.loc[mask, 'yield_per_hectare'] * 10 *
    merged.loc[mask, 'modal_price_per_quintal'] * 0.55
).round(2)

print(f"After cost merge: {merged.shape}")

# ── CALCULATE PROFIT PER HECTARE ──────────────────────────────────────────────
# Revenue per hectare = yield(tons) * 10(quintals/ton) * price(₹/quintal)
# Profit per hectare  = Revenue - Cost

merged['revenue_per_hectare'] = (
    merged['yield_per_hectare'] * 10 * merged['modal_price_per_quintal']
).round(2)

merged['profit_per_hectare'] = (
    merged['revenue_per_hectare'] - merged['cost_per_hectare']
).round(2)

merged.dropna(subset=['profit_per_hectare'], inplace=True)

print(f"\n✅ Base reference table : {merged.shape}")
print(f"   Crops   : {merged['crop'].nunique()}")
print(f"   States  : {merged['state'].nunique()}")
print(f"   Seasons : {merged['season'].nunique()}")
print(f"\nSample:")
print(merged[['crop', 'state', 'season', 'yield_per_hectare',
              'modal_price_per_quintal', 'cost_per_hectare',
              'profit_per_hectare']].head(8))

# ── SIMULATE REALISTIC FARMER ROWS FOR TRAINING ───────────────────────────────
# Each reference row = one real crop+state+season combination
# Simulate 100 different farmer scenarios per row (different land sizes)
print("\nSimulating farmer-scale training data...")

np.random.seed(42)
rows = []

for _, ref in merged.iterrows():
    for _ in range(100):
        area       = round(np.random.uniform(0.5, 5.0), 2)
        price_var  = ref['modal_price_per_quintal'] * np.random.uniform(0.88, 1.12)
        yield_var  = ref['yield_per_hectare']       * np.random.uniform(0.88, 1.12)
        cost_var   = ref['cost_per_hectare']        * np.random.uniform(0.95, 1.05)

        revenue    = yield_var * 10 * price_var * area
        total_cost = cost_var * area
        profit     = revenue - total_cost

        rows.append({
            'crop':              ref['crop'],
            'season':            ref['season'],
            'state':             ref['state'],
            'area':              area,
            'yield_per_hectare': round(yield_var, 4),
            'price_per_quintal': round(price_var, 2),
            'cost_per_hectare':  round(cost_var, 2),
            'profit':            round(profit, 2),
        })

final_df = pd.DataFrame(rows)

# Remove extreme outliers (top and bottom 3%)
q_low  = final_df['profit'].quantile(0.03)
q_high = final_df['profit'].quantile(0.97)
final_df = final_df[final_df['profit'].between(q_low, q_high)]

print(f"✅ Final training dataset : {final_df.shape}")
print(f"   Profit range          : ₹{final_df['profit'].min():,.0f} → ₹{final_df['profit'].max():,.0f}")
print(f"   Area range            : {final_df['area'].min()} → {final_df['area'].max()} ha")
print(f"\nSample rows:")
print(final_df[['crop', 'state', 'season', 'area',
                'yield_per_hectare', 'price_per_quintal',
                'cost_per_hectare', 'profit']].head(8))

# ── SAVE ──────────────────────────────────────────────────────────────────────
out = os.path.join(DATA_DIR, 'processed_data.csv')
final_df.to_csv(out, index=False)
print(f"\n✅ Saved processed_data.csv  → {out}")

ref_out = os.path.join(DATA_DIR, 'reference_table.csv')
merged.to_csv(ref_out, index=False)
print(f"✅ Saved reference_table.csv → {ref_out}")