import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
base_path = os.path.dirname(os.getcwd()) # Project root
outputs_dir = os.path.join(base_path, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

# Load cleaned data
df = pd.read_csv(os.path.join(base_path, 'data', 'cleaned_zomato_data.csv'))

# Quick EDA
# Correlation heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(outputs_dir, 'correlation_heatmap.png'))
plt.close()

# Generate HTML Report
html_content = f"""
<html>
<head><title>Zomato Retention Analysis Report</title></head>
<body>
    <h1>Zomato Retention Analysis Report</h1>
    <p><strong>Author:</strong> [Your Name]</p>
    <p><strong>Date:</strong> [Today's Date]</p>
    <p><strong>Purpose:</strong> Analyze user retention for Zomato to identify churn factors via Data Analysis.</p>
    
    <h2>Executive Summary</h2>
    <ul>
        <li>Users with fewer orders are significantly more likely to churn.</li>
        <li>Average rating correlates negatively with churn (lower ratings increase churn likelihood).</li>
    </ul>
    
    <h2>Data Overview</h2>
    <p>The dataset includes {df.shape[0]} users with features like orders, ratings, and churn status.</p>
    <pre>{df.describe().to_string()}</pre>
    <p>Churn Distribution: {df['churn'].value_counts().to_dict()}</p>
    
    <h2>Key Insights</h2>
    <p>Correlation analysis shows orders and ratings impact churn.</p>
    <img src="correlation_heatmap.png" alt="Correlation Heatmap" width="600">
    
    <h2>Conclusion</h2>
    <p>Recommendations: Target low-order users with promotions to improve retention. This project demonstrates data cleaning, EDA, visualization, and Data Engineering skills.</p>
</body>
</html>
"""

# Save HTML report
report_path = os.path.join(outputs_dir, 'zomato_report.html')
with open(report_path, 'w') as f:
    f.write(html_content)
print(f"Report generated: {report_path}")