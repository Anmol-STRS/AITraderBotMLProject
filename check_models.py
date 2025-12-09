"""Check what models are in the database and their performance"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('model_results.db')

# Get all models with their metrics
query = """
SELECT
    m.model_id,
    m.model_name,
    m.model_type,
    m.symbol,
    tr.test_r2,
    tr.test_rmse,
    tr.test_mae,
    tr.test_mape,
    tr.test_direction_accuracy,
    m.created_at
FROM models m
LEFT JOIN training_results tr ON m.model_id = tr.model_id
ORDER BY m.symbol, m.model_type, m.created_at DESC
"""

df = pd.read_sql_query(query, conn)
conn.close()

print("="*80)
print(f"Total models in database: {len(df)}")
print("="*80)

print("\nModel types breakdown:")
print(df['model_type'].value_counts())

print("\n" + "="*80)
print("XGBoost Models (These show on dashboard):")
print("="*80)
xgb_df = df[df['model_type'] == 'XGBoost'].copy()
if not xgb_df.empty:
    print(xgb_df[['symbol', 'model_name', 'test_r2', 'test_rmse', 'test_mae', 'test_direction_accuracy', 'created_at']])
    print(f"\nAverage XGBoost R²: {xgb_df['test_r2'].mean():.4f}")
    print(f"Average XGBoost RMSE: {xgb_df['test_rmse'].mean():.2f}")
else:
    print("NO XGBOOST MODELS FOUND!")

print("\n" + "="*80)
print("All models by symbol:")
print("="*80)
for symbol in sorted(df['symbol'].unique()):
    symbol_models = df[df['symbol'] == symbol]
    print(f"\n{symbol}:")
    for _, row in symbol_models.iterrows():
        r2 = f"R²={row['test_r2']:.4f}" if pd.notna(row['test_r2']) else "R²=N/A"
        print(f"  - {row['model_type']:10s} {row['model_name']:30s} {r2} (created: {row['created_at']})")

print("\n" + "="*80)
print("Checking for improved_trainer or multi_trainer models...")
print("="*80)
improved_models = df[df['model_name'].str.contains('improved', case=False, na=False)]
multi_models = df[df['model_name'].str.contains('multi', case=False, na=False)]

if not improved_models.empty:
    print("Found improved trainer models:")
    print(improved_models[['symbol', 'model_name', 'test_r2']])
else:
    print("NO improved trainer models found in database")

if not multi_models.empty:
    print("\nFound multi trainer models:")
    print(multi_models[['symbol', 'model_name', 'test_r2']])
else:
    print("NO multi trainer models found in database")
