# ⚡ Power Price Forecasting 

This project aims to forecast **Market Clearing Price (MCP)** in the Indian Energy Exchange (IEX) using machine learning, specifically **XGBoost Regressor**. Accurate MCP prediction helps reduce financial risks and optimize trading strategies in the volatile electricity market.

---

## 🧠 Problem Statement

In real-time power trading, price volatility and demand-supply imbalances create financial risks for market participants.

- **Business Goal**: Minimize financial and procurement costs.
- **Constraints**: Maximize trading returns.
- **Success Criteria**:
  - ML: Achieve **MAPE < 10%**
  - Business: 15–20% reduction in risk
  - Economic: 5–10% cost savings and 10–15% profit improvement

---

## 📁 Dataset

- **File**: `RTM_IEX_April2022_to_June2025_CSV.csv`
- **Size**: 1161 rows × 6 columns
- **Columns**:
  - `Date`
  - `Purchase Bid`
  - `Sell Bid`
  - `MCV` (Market Clearing Volume)
  - `Final Volume`
  - `MCP` (Market Clearing Price - Target)

---

## 🔍 Project Workflow

### ✅ 1. Data Exploration
- Loaded dataset using Pandas
- Stored and retrieved using MySQL + SQLAlchemy
- Visualized trends, outliers, and demand-supply relationships

### 🧱 2. Feature Engineering
- Time-based: Month, DayOfWeek, IsWeekend, WeekOfYear
- Lag features: MCP_lag1, MCP_lag7
- Rolling averages: MCP_roll7
- Volatility: MCP_vol30
- Outlier handling: Winsorization
- Scaling: MinMaxScaler

### 🧪 3. Modeling
- Tried **ARIMA/SARIMA**: High error (MAPE ~30%)
- Switched to **XGBoost**: Achieved **MAPE ~10%**
- Metrics: MAE, RMSE, MAPE

### 📊 4. Visualization
- Actual vs Predicted MCP
- Feature Importance using XGBoost
- Rolling average trend charts

### 💾 5. Deployment-Ready
- Scaled & exported dataset for Power BI
- Model & scaler saved for future use

---

## 📈 Results

| Metric  | Value     |
|---------|-----------|
| MAE     | ~0.3      |
| RMSE    | ~0.5      |
| **MAPE**| **~10% ✅**|

✅ Successfully met ML and business objectives

---

## 📦 Tools & Libraries

- Python, Pandas, Numpy, Seaborn, Matplotlib
- XGBoost, Scikit-learn, Sweetviz
- SQLAlchemy, MySQL
- Power BI (for dashboard integration)

---

## 📌 Future Improvements

- Integrate weather data and holidays
- Try deep learning models (LSTM)
- Automate retraining for real-time deployment

---

## 📬 Contact

**Author**:  K Mahesh Kumar 
📧 *manishsuryavanshi524@gmail.com  
🌐 *[LinkedIn Profile](www.linkedin.com/in/mahesh-kumar-27051m)*  

---

## 🏁 Project Status

✅ Completed initial version  
🔁 Improvements & hyperparameter tuning planned  
