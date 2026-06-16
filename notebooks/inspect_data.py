import pandas as pd
import os

DATA_DIR = r"C:\Users\aliii\OneDrive\Desktop\Yeild2Profit\data"

for fname in ['mandi_prices_real.csv', 'crop_yield_real.csv', 'cost_real.csv']:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n=== {fname} ===")
        print(f"Shape   : {df.shape}")
        print(f"Columns : {df.columns.tolist()}")
        print(df.head(3))
    else:
        print(f"\n❌ {fname} not found")