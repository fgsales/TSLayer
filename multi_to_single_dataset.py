import pandas as pd
import os

# Load the dataset
dataset = "torneo"
data = pd.read_csv(f'data/processed/{dataset}/data.csv')

# Select columns 'target' and 'year' or 'date' if they exist
target_columns = [col for col in data.columns if 'target' in col]

if 'year' in data.columns:
    target_columns.append('year')

# Filter the dataset to keep only the desired columns
filtered_data = data[target_columns]

if not os.path.exists(f'processed/{dataset}_single'):
    os.makedirs(f'data/processed/{dataset}_single')

# Save the filtered dataset
filtered_data.to_csv(f'data/processed/{dataset}_single/data.csv', index=False)