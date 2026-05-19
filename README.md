# Zomato Retention Analytics Pipeline 📊🍕

An end-to-end Data Engineering and Business Intelligence pipeline targeting user retention. We ingest raw customer data, perform Exploratory Data Analysis (EDA), transform it into a **Star Schema data model**, and serve the insights through a **Microsoft Power BI** executive dashboard and a React/FastAPI web application.

**Perfectly suited for showcasing Business Intelligence, Data Engineering, and Data Analysis skills.**

---

## 🎯 Project Highlights

Instead of standard flat-file analysis, this project demonstrates enterprise-level data patterns:
- **Data Engineering / ETL**: Python scripts automate the extraction of transactional data and perform data cleaning/transformation.
- **Kimball Data Modeling**: The output is structured into a rigorous Star Schema (`Dim_Users`, `Fact_Activity`) optimized for Power BI.
- **Exploratory Data Analysis (EDA)**: Generation of correlation matrices and churn distribution reports to identify key drivers of customer retention.
- **Advanced DAX & BI**: Utilizes calculated measures and context-aware DAX for deep drill-down analytics in Power BI.
- **API Microservice & Dashboard**: A FastAPI backend that serves the data aggregations to a responsive React dashboard, simulating an enterprise analytics gateway.

---

## 🏗️ Architecture

```text
[ Raw Data Source ] ---> [ Python ETL & Cleaning ]
                                 |
                                 V
                       [ Star Schema Data Mart ]
                       - Dim_Users 
                       - Fact_Activity
                                 |
                                 V
                      [ Backend Analytics API ] <---> [ React Dashboard ]
                      (FastAPI, Serves JSON)          (Visualizes Metrics)
                                 |
                                 V
                      [ Power BI Dashboard ]
                      (DAX, Relationships, Visuals)
```

---

## 📊 The Power BI Implementation

To impress BI hiring managers, this project goes beyond simple visualizations. Check out the dedicated guide in the `powerbi/` folder.

**Skills Demonstrated:**
1. **Data Modeling**: Building 1-to-Many relationships bridging Dimension and Fact tables.
2. **DAX Formulation**: Writing explicit measures for `Churn Rate %`, `Retained User Counts`, and conditional formatting logic.
3. **Actionable BI**: Providing actionable insights by segmenting users based on order volume and ratings.

> 👉 **View the Power BI Setup Guide:** [`powerbi/README_powerbi.md`](powerbi/README_powerbi.md)

---

## 💻 Tech Stack

- **Business Intelligence**: Power BI Desktop, DAX, Data Modeling (Star Schema)
- **Data Engineering / ETL**: Python, Pandas, Numpy
- **Backend APIs**: FastAPI, Uvicorn, Docker
- **Frontend Dashboard**: React + Vite, Recharts, Tailwind CSS

---

## 🚀 How to Run the Project

### 1. Run the ETL & BI Extract Pipeline
Clean the data and generate the Star Schema data tables locally.
```bash
pip install -r backend/requirements.txt
cd src
python data_cleaning.py
python eda.py
python export_powerbi.py
```
*Output will drop properly formatted CSVs into `data/powerbi_model/`*

### 2. Open the Power BI Dashboard
1. Import the generated Star Schema files into Power BI.
2. Follow the model structuring and DAX instructions located in the `powerbi/` directory.

### 3. Spin up the Analytics API and Dashboard
**Backend (FastAPI):**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
API Documentation available at: `http://localhost:8000/docs`

**Frontend (React):**
```bash
cd frontend
npm install
npm run dev
```

---
*Created for showcasing Data Analysis and Business Intelligence expertise.*
