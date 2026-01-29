import pandas as pd
from pathlib import Path
from datasets import load_dataset

def load_and_prepare_data():
    """Load source datasets and return a store-level pre/post outcome table with covariates and lat/lon."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    file_path = data_dir / "Store_Dataset.csv"
    store_df = pd.read_csv(file_path)

    file_path = data_dir / "store_location_dataset.csv"
    store_loc_df = pd.read_csv(file_path)
    store_loc_df['store_id'] = store_loc_df['index']

    fresh_ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    df = fresh_ds['train'].to_pandas()
    df['dt'] = pd.to_datetime(df['dt'])

    daily_metrics = df.groupby(['store_id', 'dt']).agg(
        avg_daily_sales=('sale_amount', 'sum'),
        avg_orders_per_day=('hours_sale', 'sum'),
        avg_inventory_level=('stock_hour6_22_cnt', 'mean')
    ).reset_index()

    treatment_date = pd.Timestamp("2024-05-10")
    daily_metrics['period'] = daily_metrics['dt'].apply(
        lambda d: 'pre' if d < treatment_date else 'post'
    )

    agg_metrics = daily_metrics.groupby(['store_id', 'period']).agg({
        'avg_daily_sales': 'mean',
        'avg_orders_per_day': 'mean',
        'avg_inventory_level': 'mean'
    }).unstack().reset_index()

    agg_metrics.columns = ['store_id',
        'sales_post', 'sales_pre',
        'orders_post', 'orders_pre',
        'inventory_post', 'inventory_pre'
    ]

    merged_df = pd.merge(agg_metrics, store_df, left_on='store_id', right_index=True)
    merged_df = pd.merge(
        merged_df,
        store_loc_df[['store_id', 'latitude', 'longitude']],
        on='store_id',
        how='left'
    )
    
    return merged_df

if __name__ == "__main__":
    merged_df = load_and_prepare_data()
    print(merged_df.head(5))