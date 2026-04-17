# Predictive BI & Retention Analytics Pipeline 📊🤖

An end-to-end Data Engineering and Business Intelligence pipeline targeting user retention. We ingest raw customer data, enrich it via a Machine Learning model (Random Forest), transform it into a **Star Schema data model**, and serve the insights through a **Microsoft Power BI** executive dashboard and a FastAPI backend.

**Perfectly suited for showcasing Business Intelligence, Data Engineering, and Predictive Analytics skills.**

---

## 🎯 Project Highlights for BI

Instead of standard flat-file analysis, this project demonstrates enterprise-level BI patterns:
- **ETL & Data Enrichment**: Python scripts automate the extraction of transactional data, processing it through an ML pipeline to append *predictive risk scores*.
- **Kimball Data Modeling**: The output is structured into a rigorous Star Schema (`Dim_Users`, `Fact_Activity`, `Fact_Predictions`) optimized for Power BI.
- **Advanced DAX**: Utilizes calculated measures and context-aware DAX for deep drill-down analytics.
- **API Microservice**: A FastAPI backend that serves the predictive model, simulating an enterprise analytics gateway.

---

## 🏗️ Architecture

```text
[ Raw Data Source ] ---> [ Python ETL & Cleaning ]
                                 |
                                 V
[ ML Model Deployment ] <---- [ Scikit-Learn Random Forest ]
      (FastAPI)                  | (Predicts churn probability)
                                 V
                       [ Star Schema Data Mart ]
                       - Dim_Users 
                       - Fact_Activity
                       - Fact_Predictions
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
2. **DAX Formulation**: Writing explicit measures for `Churn Risk %`, `At-Risk User Counts`, and conditional formatting logic.
3. **Actionable BI**: The dashboard isn't just descriptive; it's prescriptive, providing a "hit-list" of high-risk users sourced directly from the ML backend.

> 👉 **View the Power BI Setup Guide:** [`powerbi/README_powerbi.md`](powerbi/README_powerbi.md)

---

## 💻 Tech Stack

- **Business Intelligence**: Power BI Desktop, DAX, Data Modeling (Star Schema)
- **Data Engineering / ETL**: Python, Pandas, Numpy
- **Machine Learning**: Scikit-Learn (Random Forest Classifier)
- **Backend APIs**: FastAPI, Uvicorn, Docker
- **Frontend (Optional)**: React + Vite web dashboard

---

## 🚀 How to Run the Project

### 1. Run the ETL & BI Extract Pipeline
Generate the Star Schema data tables locally. This runs the ML model and enriches your data.
```bash
pip install -r backend/requirements.txt
cd src
python export_powerbi.py
```
*Output will drop properly formatted CSVs into `data/powerbi_model/`*

### 2. Open the Power BI Dashboard
1. Import the generated Star Schema files into Power BI.
2. Follow the model structuring and DAX instructions located in the `powerbi/` directory.

### 3. Spin up the Analytics API (Optional)
Serve your ML model locally as a RESTful service.
```bash
uvicorn backend.main:app --reload --port 8000
```
API Documentation available at: `http://localhost:8000/docs`

---
*Created by [Supriya0706](https://github.com/Supriya0706)*
