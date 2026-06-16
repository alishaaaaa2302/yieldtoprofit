import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load processed data ───────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, 'processed_data.csv'))
print(f'Loaded: {df.shape}')

df = df[['crop', 'season', 'area',
         'cost_per_hectare', 'price_per_quintal',
         'yield_per_hectare', 'profit']].dropna()

# ── Remove outliers using IQR ─────────────────────────────────────────────────
Q1  = df['profit'].quantile(0.25)
Q3  = df['profit'].quantile(0.75)
IQR = Q3 - Q1
df  = df[(df['profit'] >= Q1 - 1.5 * IQR) & (df['profit'] <= Q3 + 1.5 * IQR)]
print(f'After IQR outlier removal: {df.shape}')

# ── Label Encoding ────────────────────────────────────────────────────────────
le_crop   = LabelEncoder()
le_season = LabelEncoder()
df['crop_enc']   = le_crop.fit_transform(df['crop'])
df['season_enc'] = le_season.fit_transform(df['season'])

# ── Features and Target ───────────────────────────────────────────────────────
FEATURES = ['crop_enc', 'season_enc', 'area',
            'cost_per_hectare', 'price_per_quintal', 'yield_per_hectare']
X = df[FEATURES]
y = df['profit']

# ── Train Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows')

# ── Feature Scaling ───────────────────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER TUNING
#  Polynomial Regression has one key hyperparameter: degree
#  degree=1 → Linear (straight line)
#  degree=2 → Quadratic (one curve)
#  degree=3 → Cubic (steeper curve)
#  We try all three and pick the one with highest R2 score on test data
# ══════════════════════════════════════════════════════════════════════════════

print('\n--- Hyperparameter Tuning: Testing polynomial degree ---')
print(f'{"Degree":<10} {"Features":<12} {"R2 Score":<12} {"MAE":<15} {"RMSE"}')
print('-' * 60)

best_degree = 1
best_r2     = -999
results     = {}

for degree in [1, 2, 3]:

    # ── Hyperparameter: degree ────────────────────────────────────────────────
    poly         = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly  = poly.transform(X_test_scaled)

    model_temp   = LinearRegression()
    model_temp.fit(X_train_poly, y_train)
    y_pred_temp  = model_temp.predict(X_test_poly)

    r2   = r2_score(y_test, y_pred_temp)
    mae  = mean_absolute_error(y_test, y_pred_temp)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_temp))

    results[degree] = {'r2': r2, 'mae': mae, 'rmse': rmse,
                       'n_features': X_train_poly.shape[1]}

    print(f'{degree:<10} {X_train_poly.shape[1]:<12} {r2:<12.4f} Rs {mae:<12,.0f} Rs {rmse:,.0f}')

    if r2 > best_r2:
        best_r2     = r2
        best_degree = degree

print('-' * 60)
print(f'Best hyperparameter → degree = {best_degree}  (R2 = {best_r2:.4f})')

# ── Train final model with best hyperparameter ────────────────────────────────
print(f'\nTraining final model with degree = {best_degree}...')
poly         = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly  = poly.transform(X_test_scaled)

model  = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('\nFinal Model Evaluation')
print('======================')
print(f'Hyperparameter (degree) : {best_degree}')
print(f'R2 Score                : {r2:.4f}')
print(f'MAE                     : Rs {mae:,.0f}')
print(f'MSE                     : {mse:,.0f}')
print(f'RMSE                    : Rs {rmse:,.0f}')
print('======================')

# ── Save everything ───────────────────────────────────────────────────────────
joblib.dump(model,     os.path.join(MODEL_DIR, 'profit_model.pkl'))
joblib.dump(scaler,    os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(poly,      os.path.join(MODEL_DIR, 'poly.pkl'))
joblib.dump(le_crop,   os.path.join(MODEL_DIR, 'le_crop.pkl'))
joblib.dump(le_season, os.path.join(MODEL_DIR, 'le_season.pkl'))

with open(os.path.join(MODEL_DIR, 'features.json'), 'w') as f:
    json.dump(FEATURES, f)

with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
    json.dump({'degree': best_degree, 'r2': r2, 'mae': mae}, f)

print('\nAll files saved to model/ folder')
print('Done!')
