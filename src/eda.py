import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure outputs directory exists
outputs_dir = os.path.join(os.path.dirname(os.getcwd()), 'outputs')  # Project root + outputs
os.makedirs(outputs_dir, exist_ok=True)  # Creates if not exists

# Load data (assuming base_path is set correctly from previous fixes)
base_path = os.path.dirname(os.getcwd())  # Project root
df = pd.read_csv(os.path.join(base_path, 'data', 'cleaned_zomato_data.csv'))

# EDA: Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(outputs_dir, 'correlation_heatmap.png'))  # Use absolute path
plt.show()  # Optional: Display the plot

# Add more EDA here (e.g., boxplot)
# sns.boxplot(x='churn', y='orders', data=df)
# plt.title('Orders vs. Churn')
# plt.savefig(os.path.join(outputs_dir, 'orders_vs_churn.png'))
# plt.show()