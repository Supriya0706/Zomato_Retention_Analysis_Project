import pandas as pd

# Load data
df = pd.read_csv('../data/zomato_data.csv')

# Basic cleaning
df.dropna(inplace=True)  # Remove missing values
df['churn'] = df['churn'].astype(int)  # Ensure churn is int

# EDA: Summary stats
print(df.describe())
print(df['churn'].value_counts())  # Churn distribution

# Save cleaned data
df.to_csv('../data/cleaned_zomato_data.csv', index=False)