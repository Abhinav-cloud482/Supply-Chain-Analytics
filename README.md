# Supply-Chain-Analytics
End-to-end supply chain analytics pipeline featuring synthetic data generation, ML-based demand forecasting, inventory optimization (Safety Stock, ROP, EOQ), delivery delay risk prediction, financial impact modeling, and executive business insights.


## Supply Chain Analytics & Inventory Optimization Pipeline

An end-to-end machine learning pipeline for supply chain analytics that combines demand forecasting, inventory optimization, logistics risk prediction, and financial impact analysis into a single workflow.

The project simulates realistic supply chain operations using synthetic data and applies predictive analytics to support data-driven inventory and logistics decisions.

## Project Overview

This project consists of five major components:

1. **Synthetic Data Generation**
   - Daily demand history
   - Shipment records
   - Supplier and carrier features
   - Weather impact simulation

2. **Demand Forecasting**
   - Random Forest Regressor
   - TimeSeriesSplit Cross Validation
   - GridSearchCV hyperparameter tuning
   - Forecast performance evaluation

3. **Inventory Optimization**
   - Safety Stock calculation
   - Reorder Point (ROP)
   - Economic Order Quantity (EOQ)
   - Financial cost simulation

4. **Delivery Delay Prediction**
   - Logistic Regression classifier
   - Feature standardization
   - Delay risk prediction
   - Identification of key operational risk drivers

5. **Executive Business Summary**
   - Inventory recommendations
   - Annual holding cost savings
   - Supply chain risk insights
   - Strategic business recommendations

## Machine Learning Models

### Demand Forecasting
- Random Forest Regressor
- TimeSeriesSplit Cross Validation
- GridSearchCV Hyperparameter Optimization

### Delivery Delay Classification
- Logistic Regression
- StandardScaler
- Binary Classification Metrics

## Evaluation Metrics

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification
- Accuracy
- Precision
- Recall
- F1 Score

## Inventory Optimization Metrics

The optimization engine computes:

- Safety Stock
- Reorder Point (ROP)
- Economic Order Quantity (EOQ)
- Service Level Target
- Annual Holding Cost
- Optimized Holding Cost
- Estimated Cost Savings

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Logging

## Project Structure

```text
.
├── supply_chain_analytics.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/supply-chain-analytics.git

cd supply-chain-analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

```text
numpy==2.3.2
pandas==2.3.2
scipy==1.16.1
scikit-learn==1.7.1
```

## Usage

Run the pipeline:

```bash
python supply_chain_analytics.py
```

## Pipeline Workflow

```text
Synthetic Data
      │
      ▼
Demand Forecasting
      │
      ▼
Inventory Optimization
      │
      ▼
Financial Cost Analysis
      │
      ▼
Delay Risk Prediction
      │
      ▼
Executive Business Summary
```

## Example Output

The pipeline reports:

- Demand forecasting performance
- Optimal inventory policy
- Safety stock recommendations
- Reorder point
- EOQ calculation
- Annual holding cost savings
- Logistics delay prediction metrics
- Most influential delay risk factors
- Executive strategic recommendations

## Business Value

This project demonstrates how machine learning can improve supply chain operations by:

- Forecasting customer demand
- Optimizing inventory levels
- Reducing holding costs
- Predicting transportation delays
- Identifying operational risks
- Supporting data-driven business decisions

## Future Improvements

- XGBoost and LightGBM forecasting models
- LSTM-based demand prediction
- Multi-product inventory optimization
- Supplier performance dashboards
- Interactive Power BI/Tableau visualizations
- Real-world ERP and warehouse data integration
- API deployment with FastAPI
- Docker containerization

## Author

**Abhinav Dixit**

GitHub: [Abhinav Dixit](https://github.com/Abhinav-cloud482)

Machine Learning • Data Science • Supply Chain Analytics
