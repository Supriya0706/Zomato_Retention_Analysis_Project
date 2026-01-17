import pandas as pd
import numpy as np


np.random.seed(42)
data = {
    'user_id': range(1, 1001),
    'orders': np.random.poisson(5, 1000),  # Average 5 orders
    'avg_rating': np.random.uniform(1, 5, 1000),
    'last_order_days': np.random.exponential(30, 1000),  # Days since last order
    'churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 0: Retained, 1: Churned
}
df = pd.DataFrame(data)
df.to_csv('../data/zomato_data.csv', index=False)
print("Synthetic data generated.")