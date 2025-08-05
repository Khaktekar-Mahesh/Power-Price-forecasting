# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 14:21:44 2025

@author: admin
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import StringIO
from datetime import datetime
from xgboost import XGBRegressor, DMatrix, train as xgb_train, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

st.title("⚡ Power Market Analysis & MCP Prediction using XGBoost")

uploaded_file = st.file_uploader("Upload the Power Market CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Data Preprocessing ---
    st.subheader("📊 Raw Data Overview")
    st.dataframe(df.head())

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    df['MCP_Rolling'] = df['MCP'].rolling(window=7).mean()
    df.dropna(subset=['MCP_Rolling'], inplace=True)

    df.set_index('Date', inplace=True)
    df = df.sort_index()
    df['Month']      = df.index.month
    df['Quarter']    = df.index.quarter
    df['DayOfWeek']  = df.index.dayofweek
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
    df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
    df['Day']        = df.index.day

    df['MCP_lag1'] = df['MCP'].shift(1)
    df['MCP_lag7'] = df['MCP'].shift(7)
    df['MCP_roll7'] = df['MCP'].rolling(7).mean()
    df['MCP_vol30'] = df['MCP'].rolling(30).std()
    df['MCP_Capped'] = winsorize(df['MCP'], limits=[0.05, 0.05])
    df.dropna(inplace=True)

    # --- Scaling ---
    scale_cols = [
        'Purchase Bid', 'Sell Bid', 'MCV', 'Final_Volume',
        'MCP_lag1', 'MCP_lag7',
        'MCP_Rolling', 'MCP_roll7',
        'MCP_vol30',
        'MCP_Capped'
    ]

    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # --- Train-Test Split ---
    target_col = 'MCP'
    y = df[target_col]
    features = [
        'Purchase Bid', 'Sell Bid', 'MCV', 'Final_Volume',
        'MCP_lag1', 'MCP_lag7', 'MCP_roll7', 'MCP_vol30', 'MCP_Capped',
        'Month', 'Quarter', 'DayOfWeek', 'IsWeekend', 'WeekOfYear', 'Day'
    ]
    X = df[features]

    cutoff_train = '2024-10-31'
    cutoff_val   = '2024-12-31'
    X_train = X.loc[:cutoff_train]
    y_train = y.loc[:cutoff_train]
    X_val = X.loc[cutoff_train:cutoff_val]
    y_val = y.loc[cutoff_train:cutoff_val]
    X_test = X.loc[cutoff_val:]
    y_test = y.loc[cutoff_val:]

    # --- XGBoost Training ---
    st.subheader("📈 Training XGBoost Regressor")
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)
    dtest = DMatrix(X_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb_train(params, dtrain, num_boost_round=100, early_stopping_rounds=10, evals=watchlist, verbose_eval=False)

    y_pred = model.predict(dtest)

    # --- Evaluation Metrics ---
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape_val = mape(y_test, y_pred)

    st.markdown(f"✅ **MAE**: {mae:.3f}")
    st.markdown(f"✅ **RMSE**: {rmse:.3f}")
    st.markdown(f"✅ **MAPE**: {mape_val:.2f}%")

    # --- Prediction vs Actual Plot ---
    st.subheader("🔍 Predicted vs Actual MCP")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, label='Actual MCP', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted MCP', color='orange')
    ax.set_title('Actual vs Predicted MCP')
    ax.set_xlabel('Date')
    ax.set_ylabel('MCP')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Feature Importance ---
    st.subheader("🔑 Feature Importances")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_importance(model, max_num_features=15, importance_type='gain', ax=ax2)
    st.pyplot(fig2)

    # --- Raw Data Preview (optional download) ---
    st.subheader("📥 Download Processed Data")
    csv_download = df.reset_index().to_csv(index=False)
    st.download_button("Download CSV", data=csv_download, file_name="processed_power_data.csv", mime="text/csv")

else:
    st.info("Please upload a valid CSV file to begin analysis.")
