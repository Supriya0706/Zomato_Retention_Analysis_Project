# Data Engineering & BI Pipeline for Power BI

To stand out for an **Associate BI Role**, your portfolio needs to show more than just dragging and dropping charts onto a canvas. It needs to show you understand **Data Modeling (Star Schemas)**, **ETL workflows**, and **Advanced DAX**.

This project simulates a real-world enterprise BI workflow: Python extracts the raw data, applies a Machine Learning model to enrich the data with predictive scores, and loads the output into a **Star Schema** optimized for Power BI.

## Step 1: Run the ETL Script
Ensure your backend dependencies are installed. Then, from the `src` folder, run:
```bash
python export_powerbi.py
```
This generates three files in `data/powerbi_model/`:
1. `Dim_Users.csv` (Demographics and User attributes)
2. `Fact_Activity.csv` (Transactional aggregates and actual churn history)
3. `Fact_Predictions.csv` (Machine Learning output / Predictive metrics)

## Step 2: Power BI Data Modeling (The "Star Schema")
1. Open Power BI Desktop.
2. Click **Get Data -> Text/CSV** and import all three files from the `data/powerbi_model/` folder.
3. Open the **Model View** (the relationship icon on the far left).
4. Drag connections to establish the following **1-to-Many** relationships:
   - `Dim_Users[user_id]` -> `Fact_Activity[user_id]`
   - `Dim_Users[user_id]` -> `Fact_Predictions[user_id]`

*Why this matters for your resume: Interviewers look for proper relationship management and understanding of Fact vs. Dimension tables.*

## Step 3: Write Advanced DAX Measures
Instead of just using implicit sums, create a **Measure Table** and write these DAX formulas:

```dax
// 1. Total Users Base
Total Users = COUNTROWS('Dim_Users')

// 2. Churn Rate %
Actual Churn Rate = DIVIDE(SUM('Fact_Activity'[actual_churn_flag]), [Total Users], 0)

// 3. High Risk Users Identified by ML
Users at High Risk = CALCULATE([Total Users], 'Fact_Predictions'[risk_segment] = "High Risk")

// 4. Average Churn Probability Score
Average Risk Score = AVERAGE('Fact_Predictions'[churn_probability_score])
```

## Step 4: Build the Dashboard
Create a 2-page report to tell a data story:

### Page 1: Executive Churn Summary (Historical Data)
- **Top KPIs**: Total Users, Actual Churn Rate, Average Orders.
- **Visuals**: 
  - Churn Rate by `Dim_Users[region]` (Map or Bar chart).
  - Orders by `Fact_Activity[avg_rating]` (Scatter plot).

### Page 2: ML Predictive Action Center
- **Top KPIs**: Users at High Risk, Average Risk Score.
- **Visuals**:
  - `Dim_Users[acquisition_channel]` vs. `Users at High Risk` (Identify which marketing channels bring in risky users).
  - Table Visual: List of `user_id`, `region`, and `churn_probability_score` sorted descending to give a "hit list" to the retention team.

## Step 5: Showcase in your Portfolio
Do not just list "Dashboard built". In your resume/GitHub, describe the workflow:
> *"Engineered an automated Python ETL pipeline to cleanse activity data, processed it through a Scikit-Learn Random Forest model, and modeled a Star Schema Data Mart. Connected Power BI to this schema utilizing custom DAX measures to highlight high-risk user segments."*
