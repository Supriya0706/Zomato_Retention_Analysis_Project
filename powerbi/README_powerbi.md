# Data Engineering & BI Pipeline for Power BI

To stand out for an **Associate BI Role**, your portfolio needs to show more than just dragging and dropping charts onto a canvas. It needs to show you understand **Data Modeling (Star Schemas)**, **ETL workflows**, and **Advanced DAX**.

This project simulates a real-world enterprise BI workflow: Python extracts the raw data, performs data cleaning and Exploratory Data Analysis, and loads the output into a **Star Schema** optimized for Power BI.

## Step 1: Run the ETL Script
Ensure your backend dependencies are installed. Then, from the `src` folder, run:
```bash
python export_powerbi.py
```
This generates two files in `data/powerbi_model/`:
1. `Dim_Users.csv` (Demographics and User attributes)
2. `Fact_Activity.csv` (Transactional aggregates and actual churn history)

## Step 2: Power BI Data Modeling (The "Star Schema")
1. Open Power BI Desktop.
2. Click **Get Data -> Text/CSV** and import both files from the `data/powerbi_model/` folder.
3. Open the **Model View** (the relationship icon on the far left).
4. Drag connections to establish the following **1-to-Many** relationship:
   - `Dim_Users[user_id]` -> `Fact_Activity[user_id]`

*Why this matters for your resume: Interviewers look for proper relationship management and understanding of Fact vs. Dimension tables.*

## Step 3: Write Advanced DAX Measures
Instead of just using implicit sums, create a **Measure Table** and write these DAX formulas:

```dax
// 1. Total Users Base
Total Users = COUNTROWS('Dim_Users')

// 2. Churn Rate %
Actual Churn Rate = DIVIDE(SUM('Fact_Activity'[actual_churn_flag]), [Total Users], 0)

// 3. Average Orders
Average Orders = AVERAGE('Fact_Activity'[orders])

// 4. Retained Users
Retained Users = CALCULATE([Total Users], 'Fact_Activity'[actual_churn_flag] = 0)
```

## Step 4: Build the Dashboard
Create an executive report to tell a data story:

### Executive Churn Summary
- **Top KPIs**: Total Users, Actual Churn Rate, Average Orders, Retained Users.
- **Visuals**: 
  - Churn Rate by `Dim_Users[region]` (Map or Bar chart).
  - Orders vs. `Fact_Activity[avg_rating]` (Scatter plot).
  - Retention by `Dim_Users[acquisition_channel]` (Bar chart).

## Step 5: Showcase in your Portfolio
Do not just list "Dashboard built". In your resume/GitHub, describe the workflow:
> *"Engineered an automated Python ETL pipeline to cleanse activity data and modeled a Star Schema Data Mart. Connected Power BI to this schema utilizing custom DAX measures to highlight user segments and perform deep-dive exploratory data analysis on churn."*
