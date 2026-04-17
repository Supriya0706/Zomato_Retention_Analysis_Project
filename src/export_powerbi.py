import pandas as pd
import numpy as np
import os
import datetime

import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
data_out_dir = os.path.join(base_path, 'data', 'powerbi_model')

# Ensure the output directory exists
os.makedirs(data_out_dir, exist_ok=True)

try:
    from backend.model import get_model_and_scaler
    # Use the unified model loading logic to guarantee consistency
    model, scaler = get_model_and_scaler()
    
    # Load dataset
    df = pd.read_csv(os.path.join(base_path, 'data', 'zomato_data.csv'))
    df.dropna(inplace=True)
    df['churn'] = df['churn'].astype(int)

    # ------------------------------------------------------------------------
    # Create Data Warehouse / BI Star Schema Extracts
    # ------------------------------------------------------------------------
    print("Generating BI Star Schema Datasets...")

    # 1. Dimension Table: Dim_Users
    # Simulate some demographic data for richer BI slicing
    np.random.seed(42)
    regions = ['North', 'South', 'East', 'West']
    acquisition_channels = ['Organic', 'Paid Social', 'Referral', 'Email']
    
    dim_users = pd.DataFrame({
        'user_id': df['user_id'],
        'region': np.random.choice(regions, size=len(df)),
        'acquisition_channel': np.random.choice(acquisition_channels, size=len(df)),
        'join_date': pd.date_range(start='2023-01-01', periods=len(df), freq='D').strftime('%Y-%m-%d')
    })
    
    # 2. Fact Table: Fact_Customer_Activity
    fact_activity = pd.DataFrame({
        'user_id': df['user_id'],
        'orders': df['orders'],
        'avg_rating': df['avg_rating'],
        'last_order_days': df['last_order_days'],
        'actual_churn_flag': df['churn']
    })
    
    # 3. Fact Table: Fact_ML_Predictions
    # Generate predictions using the loaded ML model
    X = df[['orders', 'avg_rating', 'last_order_days']]
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[:, 1]
    
    fact_predictions = pd.DataFrame({
        'user_id': df['user_id'],
        'churn_probability_score': (probs * 100).round(2),
        'model_run_date': datetime.date.today().strftime('%Y-%m-%d'),
    })
    
    # Add BI specific calculated dimensions
    fact_predictions['risk_segment'] = pd.cut(
        fact_predictions['churn_probability_score'], 
        bins=[-1, 30, 70, 100], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    ).astype(str)

    # Export to CSV
    dim_users.to_csv(os.path.join(data_out_dir, 'Dim_Users.csv'), index=False)
    fact_activity.to_csv(os.path.join(data_out_dir, 'Fact_Activity.csv'), index=False)
    fact_predictions.to_csv(os.path.join(data_out_dir, 'Fact_Predictions.csv'), index=False)
    
    print(f"✅ Successfully generated Star Schema exports in: {data_out_dir}")
    print("- Dim_Users.csv")
    print("- Fact_Activity.csv")
    print("- Fact_Predictions.csv")

except Exception as e:
    print(f"Failed to generate Power BI export: {e}. Ensure the backend model runs first.")
