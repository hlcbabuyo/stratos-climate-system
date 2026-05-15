# STRATOS: Spatial Thermal Risk & Temperature Observation System

![STRATOS Banner](https://via.placeholder.com/1200x300.png?text=STRATOS:+Spatial+Thermal+Risk+&+Temperature+Observation+System)

## Overview
STRATOS is a dual-purpose Machine Learning portfolio project and academic submission. It focuses on building a machine learning pipeline to analyze climate anomalies and extreme heat events in Misamis Oriental, Philippines. 

The project seamlessly integrates automated data engineering, advanced machine learning (supervised, unsupervised, and reinforcement learning), and production deployment via a RESTful API.

## Repository Structure
```
stratos-climate-system/
├── app/
│   ├── main.py                  # FastAPI server to deploy the winning model
│   └── model.pkl                # The trained algorithm exported from Colab
├── data/
│   ├── raw/                     # Raw downloaded CSV data
│   └── processed/               # Cleaned data (if applicable)
├── notebooks/
│   └── 01_stratos_ml_pipeline.ipynb  # The core academic submission
├── scripts/
│   └── etl_nasa_power.py        # Automated data engineering script
├── requirements.txt
└── README.md                    
```

## Phase 1: ETL Pipeline
The data engineering process utilizes the NASA POWER API to automatically fetch 10 years of daily meteorological data (Temperature, Humidity, Solar Radiation, Wind Speed) for Misamis Oriental.
- **Run the ETL Script:**
  ```bash
  python scripts/etl_nasa_power.py
  ```
- *Output:* Generates `data/raw/stratos_climate_10yr.csv` containing over 3,600 daily records.

## Phase 2-4: The Academic Core (13 Algorithms & Concepts)
The core analysis is located in `notebooks/01_stratos_ml_pipeline.ipynb`, built specifically for Google Colab execution. The notebook fulfills all academic requirements by implementing exactly 13 algorithms and ML concepts:

**Preprocessing & Unsupervised Learning**
1. **Covariance Analysis & Z-Score Normalization** (Prerequisites)
2. **Principal Component Analysis (PCA)** - Dimensionality reduction.
3. **K-Means Algorithm** - Clustering days into distinct climate profiles.
4. **Linear Regression** - Predicting peak temperatures.
5. **Gradient Descent** - Optimizing regression weights from scratch.

**Classification Showdown**
Engineered a binary target (`extreme_heat_warning`) and evaluated:
6. **Logistic Regression**
7. **Decision Trees**
8. **Random Forest** (Best Performing Model)
9. **Support Vector Machine (SVM) Algorithm** (Base Implementation)
10. **Linear SVM**
11. **Non-Linear SVM** (Polynomial)
12. **Kernel Trick in SVM** (Mathematical Transformation Demonstration)
13. **RBF Kernel in SVM**

**Reinforcement Learning**
14. **Q-Learning** (Bonus concept numbering adjustment): Using the Bellman Equation to find optimal delivery paths across a municipal grid, avoiding predicted extreme heat zones.

## Phase 5: Production Deployment
The best performing model (Random Forest) is exported as `model.pkl` and served using a FastAPI backend to provide real-time extreme heat predictions.

### How to Run the FastAPI Server

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure Model Exists:** Make sure you have run the Colab Notebook to generate `app/model.pkl`.
3. **Start the Server:**
   ```bash
   cd app
   uvicorn main:app --reload
   ```
4. **Access the API:**
   - Interactive Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Prediction Endpoint: `POST /predict`
   
   **Sample JSON Payload:**
   ```json
   {
       "temperature_2m": 35.5,
       "relative_humidity_2m": 65.0,
       "solar_radiation": 20.1,
       "wind_speed_10m": 4.5
   }
   ```