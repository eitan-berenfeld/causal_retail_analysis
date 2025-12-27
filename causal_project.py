import pandas as pd

file_path = "my_project/data/Store_Dataset.csv"
store_df = pd.read_csv(file_path)

file_path = "my_project/data/store_location_dataset.csv"
store_loc_df = pd.read_csv(file_path)
store_loc_df['store_id'] = store_loc_df['index']

from datasets import load_dataset

fresh_ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
df = fresh_df['train'].to_pandas()
df.head()
df.columns

