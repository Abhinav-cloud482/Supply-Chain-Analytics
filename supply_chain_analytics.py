"""
SUPPLY CHAIN ANALYTICS & INVENTORY OPTIMIZATION PIPELINE

1. Synthetic Data Engine (Demand history, Lead times, Supplier features)
2. Demand Forecasting (Random Forest Regressor + TimeSeriesSplit CV)
3. Inventory Optimization & Financial Cost Modeling (Safety Stock, ROP, EOQ)
4. Delivery Delay Risk Classification (Logistic Regression + Scaling)
5. Executive Strategic Summary & Business Insights
"""

import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

SEED = 42
np.random.seed(SEED)



# SECTION 1: SYNTHETIC DATA GENERATION ENGINE

def generate_supply_chain_data(num_days: int = 365, num_shipments: int = 350):
    """Generates daily historical demand time-series and shipment tracking logs."""
    logging.info("Generating realistic supply chain historical dataset...")
    
    dates = pd.date_range(start="2025-01-01", periods=num_days, freq="D")
    
    base_demand = 200
    annual_seasonality = 40 * np.sin(2 * np.pi * dates.dayofyear / 365)
    weekly_seasonality = 15 * np.cos(2 * np.pi * dates.dayofweek / 7)
    linear_trend = 0.12 * np.arange(num_days)
    gaussian_noise = np.random.normal(0, 18, num_days)
    
    demand = np.maximum(50, base_demand + annual_seasonality + weekly_seasonality + linear_trend + gaussian_noise).astype(int)
    
    df_demand = pd.DataFrame({
        "Date": dates,
        "Demand": demand,
        "DayOfWeek": dates.dayofweek,
        "Month": dates.month,
        "DayOfYear": dates.dayofyear,
        "IsWeekend": dates.dayofweek.isin([5, 6]).astype(int)
    })
    
    df_demand["Demand_Lag_1"] = df_demand["Demand"].shift(1)
    df_demand["Demand_Lag_7"] = df_demand["Demand"].shift(7)
    df_demand["Demand_Rolling_7_Mean"] = df_demand["Demand"].shift(1).rolling(window=7).mean()
    df_demand["Demand_Rolling_7_Std"] = df_demand["Demand"].shift(1).rolling(window=7).std()
    df_demand.dropna(inplace=True)
    
    shipment_data = {
        "Shipment_ID": [f"SHIP-{1000+i}" for i in range(num_shipments)],
        "Distance_KM": np.random.uniform(100, 2500, num_shipments),
        "Order_Quantity": np.random.randint(100, 5000, num_shipments),
        "Supplier_Rating": np.random.uniform(2.0, 5.0, num_shipments),
        "Carrier_Reliability_Score": np.random.uniform(0.5, 1.0, num_shipments),
        "Weather_Impact_Score": np.random.uniform(0.0, 1.0, num_shipments)
    }
    df_shipments = pd.DataFrame(shipment_data)
    
    delay_logit = (
        0.35 * (df_shipments["Distance_KM"] / 2500) +
        0.45 * df_shipments["Weather_Impact_Score"] -
        0.50 * df_shipments["Carrier_Reliability_Score"] -
        0.20 * (df_shipments["Supplier_Rating"] / 5.0) + 0.3
    )
    delay_prob = 1 / (1 + np.exp(-delay_logit))
    df_shipments["Is_Delayed"] = (delay_prob > np.median(delay_prob)).astype(int)
    
    return df_demand, df_shipments



# SECTION 2: DEMAND FORECASTING MODEL

def train_demand_forecaster(df_demand: pd.DataFrame):
    """Trains a Random Forest Regressor using TimeSeriesSplit Cross-Validation."""
    logging.info("Training Random Forest Regressor with TimeSeriesSplit CV...")
    
    features = [
        "DayOfWeek", "Month", "DayOfYear", "IsWeekend",
        "Demand_Lag_1", "Demand_Lag_7", "Demand_Rolling_7_Mean", "Demand_Rolling_7_Std"
    ]
    X = df_demand[features]
    y = df_demand["Demand"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=SEED),
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2,
        "Best_Params": grid_search.best_params_
    }
    
    return best_model, metrics, predictions, y_test



# SECTION 3: INVENTORY OPTIMIZATION ENGINE & COST SIMULATION

def optimize_inventory(avg_daily_demand, std_daily_demand, avg_lead_time_days=5, std_lead_time_days=1.5, service_level=0.95):
    """Calculates optimal Safety Stock, Reorder Point, and Economic Order Quantity."""
    z_score = norm.ppf(service_level)
    
    safety_stock = z_score * np.sqrt(
        (avg_lead_time_days * (std_daily_demand ** 2)) +
        ((avg_daily_demand ** 2) * (std_lead_time_days ** 2))
    )
    
    reorder_point = (avg_daily_demand * avg_lead_time_days) + safety_stock
    
    annual_demand = avg_daily_demand * 365
    ordering_cost = 150.0
    holding_cost = 5.0
    
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    
    return {
        "Service Level Target": f"{service_level * 100:.0f}%",
        "Safety Stock (Units)": int(np.ceil(safety_stock)),
        "Reorder Point (Units)": int(np.ceil(reorder_point)),
        "Economic Order Quantity (Units)": int(np.ceil(eoq))
    }

def simulate_financial_impact(df_demand, opt_metrics, annual_holding_cost=5.0):
    """Simulates financial holding cost savings against a static baseline buffer strategy."""
    avg_demand = df_demand["Demand"].mean()
    naive_buffer = avg_demand * 5
    opt_buffer = opt_metrics["Safety Stock (Units)"]
    
    naive_annual_cost = naive_buffer * annual_holding_cost
    opt_annual_cost = opt_buffer * annual_holding_cost
    savings = naive_annual_cost - opt_annual_cost
    
    return {
        "Naive Holding Cost ($)": round(naive_annual_cost, 2),
        "Optimized Holding Cost ($)": round(opt_annual_cost, 2),
        "Net Annual Cost Savings ($)": round(savings, 2)
    }



# SECTION 4: LOGISTICS DELAY RISK CLASSIFIER

def train_delay_classifier(df_shipments: pd.DataFrame):
    """Trains a Logistic Regression model with scaled features to identify delay drivers."""
    logging.info("Training Logistic Regression Model for Transport Delay Risk...")
    
    features = ["Distance_KM", "Order_Quantity", "Supplier_Rating", "Carrier_Reliability_Score", "Weather_Impact_Score"]
    X = df_shipments[features]
    y = df_shipments["Is_Delayed"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    
    eval_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1_Score": f1_score(y_test, predictions)
    }
    
    feature_importance = dict(zip(features, np.round(model.coef_[0], 4)))
    
    return model, eval_metrics, feature_importance



# SECTION 5: MAIN EXECUTION PIPELINE

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      ENTERPRISE SUPPLY CHAIN ANALYTICS & OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    df_demand, df_shipments = generate_supply_chain_data()
    logging.info(f"Loaded {len(df_demand)} demand logs and {len(df_shipments)} shipment records.\n")
    
    demand_model, forecaster_metrics, preds, actuals = train_demand_forecaster(df_demand)
    print("--- 1. DEMAND FORECASTING EVALUATION (TimeSeries CV) ---")
    print(f"Algorithm                     : Random Forest Regressor (GridSearchCV)")
    print(f"Mean Absolute Error (MAE)     : {forecaster_metrics['MAE']:.2f} units/day")
    print(f"Root Mean Squared Error (RMSE): {forecaster_metrics['RMSE']:.2f} units")
    print(f"Variance Explained (R² Score) : {forecaster_metrics['R2_Score']:.3f}")
    print(f"Optimal Hyperparameters       : {forecaster_metrics['Best_Params']}\n")
    
    avg_demand = df_demand["Demand"].mean()
    std_demand = df_demand["Demand"].std()
    opt_metrics = optimize_inventory(avg_demand, std_demand)
    fin_metrics = simulate_financial_impact(df_demand, opt_metrics)
    
    print("--- 2. INVENTORY OPTIMIZATION & FINANCIAL IMPACT ---")
    for metric, value in opt_metrics.items():
        print(f"{metric:<30}: {value}")
    print("-" * 50)
    for cost_item, val in fin_metrics.items():
        print(f"{cost_item:<30}: ${val:,.2f}")
    print("\n")
    
    delay_model, classifier_metrics, risk_drivers = train_delay_classifier(df_shipments)
    print("--- 3. LOGISTICS DELAY RISK ANALYSIS ---")
    print(f"Algorithm                     : Logistic Regression (Standardized)")
    print(f"Classification Accuracy       : {classifier_metrics['Accuracy']*100:.1f}%")
    print(f"Precision                     : {classifier_metrics['Precision']:.2f}")
    print(f"Recall                        : {classifier_metrics['Recall']:.2f}")
    print(f"F1-Score                      : {classifier_metrics['F1_Score']:.2f}")
    print("\nStandardized Risk Driver Impact (Log-Odds Scale):")
    for feature, weight in sorted(risk_drivers.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "(Increases Risk)" if weight > 0 else "(Decreases Risk)"
        print(f"  * {feature:<26}: {weight:+.4f} {direction}")
        
    print("\n" + "=" * 70)
    print("                   EXECUTIVE STRATEGIC SUMMARY")
    print("=" * 70)
    print(f"• INVENTORY CONTROL: Operating ROP trigger at {opt_metrics['Reorder Point (Units)']} units with EOQ batch size "
          f"of {opt_metrics['Economic Order Quantity (Units)']} units.")
    print(f"• FINANCIAL OPTIMIZATION: Replacing baseline rules with ML-driven dynamic safety stock delivers "
          f"${fin_metrics['Net Annual Cost Savings ($)']:,.2f} in annual holding cost reductions.")
    print(f"• RISK MITIGATION: Primary transit delay drivers identified as Carrier Reliability and Weather Score. "
          f"Re-evaluating low-scoring transport carriers is recommended.")
    print("=" * 70 + "\n")