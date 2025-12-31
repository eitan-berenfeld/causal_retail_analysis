import pandas as pd

file_path = "data/Store_Dataset.csv"
store_df = pd.read_csv(file_path)

file_path = "data/store_location_dataset.csv"
store_loc_df = pd.read_csv(file_path)
store_loc_df['store_id'] = store_loc_df['index']

from datasets import load_dataset

fresh_ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
df = fresh_ds['train'].to_pandas()

df['dt'] = pd.to_datetime(df['dt'])

# Aggregate to daily store-level metrics
daily_metrics = df.groupby(['store_id', 'dt']).agg(
    avg_daily_sales=('sale_amount', 'sum'),
    avg_orders_per_day=('hours_sale', 'sum'),
    avg_inventory_level=('stock_hour6_22_cnt', 'mean')
).reset_index()

# Choose a treatment date (e.g., middle of dataset)
treatment_date = pd.Timestamp("2024-05-10")

# Label pre/post period
daily_metrics['period'] = daily_metrics['dt'].apply(
    lambda d: 'pre' if d < treatment_date else 'post'
)

# Compute pre/post average metrics per store
agg_metrics = daily_metrics.groupby(['store_id', 'period']).agg({
    'avg_daily_sales': 'mean',
    'avg_orders_per_day': 'mean',
    'avg_inventory_level': 'mean'
}).unstack().reset_index()

# Flatten multi-index columns
agg_metrics.columns = ['store_id',
    'sales_post', 'sales_pre',
    'orders_post', 'orders_pre',
    'inventory_post', 'inventory_pre'
]

# np.random.seed(42)
# agg_metrics['treated'] = np.random.choice([0, 1], size=len(agg_metrics))

merged_df = pd.merge(agg_metrics, store_df, left_on='store_id', right_index=True)
# Merge lat/lon into merged_df using store_id
df = pd.merge(
    merged_df,
    store_loc_df[['store_id', 'latitude', 'longitude']],
    on='store_id',
    how='left'
)
print(df.head(5))