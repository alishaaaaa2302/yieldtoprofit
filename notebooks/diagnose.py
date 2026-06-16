import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'processed_data.csv'))

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\n--- area ---")
print(df['area'].describe())
print("\n--- yield_per_hectare ---")
print(df['yield_per_hectare'].describe())
print("\n--- profit ---")
print(df['profit'].describe())
print("\n--- Sample rows ---")
print(df[['crop','season','area','yield_per_hectare',
          'cost_per_hectare','price_per_quintal','profit']].head(10))