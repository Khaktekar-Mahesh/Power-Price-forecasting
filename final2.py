# -*- coding: utf-8 -*-
"""

Project 1
Topic: Power Price Prediction
Project_Group_273
"""

# Data Loading
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sweetviz as sv
from sqlalchemy import create_engine
from urllib.parse import quote

user = 'root'
password = quote('1234567890')
db = 'air'
# Connect to MySQL database 'wall_db' using credentials and create an engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")

file_path = r"C:/Users/admin/Downloads/RTM_IEX_April2022 _to_June2025_CSV.csv"
df = pd.read_csv(file_path)

df.to_sql('power', con=engine, if_exists='replace', chunksize=1000, index=False)
sql = 'select * from power'
df = pd.read_sql_query(sql, con=engine)

print(df.shape)   # (1161, 6)

print(df.columns)
#(['Date', 'Purchase Bid', 'Sell Bid', 'MCV', 'Final_Volume', 'MCP'], dtype='object')

print(df.dtypes)
                  
print(df.head())  # Prints First 5 Columns

report = sv.analyze(df)
report.show_html('report.html')
#Date Handling
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y') #formatting date to dd-mm-yyyy

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month     #extracting month from date field
df['DayOfWeek'] = df['Date'].dt.day_name()   #extracting day of the week from date field

df.isnull().sum() #checking missing values
df.duplicated().sum()  #Checking duplicate values

df.describe()
df.describe(include='object')

sns.histplot(df['MCP'], kde=True)
sns.boxplot(x=df['MCP'])

# Ploting MCP over time to identify the outliers

plt.figure(figsize=(14, 6))
sns.lineplot(x='Date', y='MCP', data=df)

plt.title('MCP Over Time')
plt.xlabel('Date')
plt.ylabel('Market Clearing Price (MCP)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plotting MCP Rolling Mean over date

df['MCP_Rolling'] = df['MCP'].rolling(window=7).mean()
sns.lineplot(x='Date', y='MCP_Rolling', data=df)

plt.xticks(rotation=45)
plt.title("7-Day Rolling Average of MCP")
plt.xlabel("Date")
plt.ylabel("MCP (Rolling Mean)")
plt.grid(True)

# Actual MCP Vs &day Rolling Average MCP

sns.lineplot(x='Date', y='MCP', data=df, label='Actual MCP', alpha=0.3)
sns.lineplot(x='Date', y='MCP_Rolling', data=df, label='7-Day Rolling Avg')

#Monthly Aggregation

df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_avg = df.groupby('YearMonth')['MCP'].mean().reset_index()
monthly_avg['YearMonth'] = monthly_avg['YearMonth'].astype(str)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.title("Monthly Average Market Clearing Price (MCP)")
plt.xlabel("Year-Month")
plt.ylabel("Average MCP (₹)")

sns.lineplot(x='YearMonth', y='MCP', data=monthly_avg)
plt.xticks(rotation=45)


#Monthly Avg Vs 3month Rolling MOnthly AVG

plt.figure(figsize=(12, 6))
# Original monthly MCP
sns.lineplot(data=monthly_avg, x='YearMonth', y='MCP', label='Monthly MCP')
# Smoothed version
monthly_avg['MCP_Smooth'] = monthly_avg['MCP'].rolling(window=3, center=True, min_periods=1).mean()


sns.lineplot(data=monthly_avg, x='YearMonth', y='MCP_Smooth', label='3-Month Rolling Avg', color='orange')
plt.xticks(rotation=45)
plt.title("Monthly MCP vs 3-Month Rolling Average")
plt.xlabel("Year-Month")
plt.ylabel("MCP")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



sns.histplot(df['MCV'], kde=True)

#Bivariate Scatter Plot

sns.scatterplot(x='MCP', y='MCV', data=df)

#Price Vs Demand

plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Date', y='MCP', label='Price')
sns.lineplot(data=df, x='Date', y='Purchase Bid', label='Demand')

#Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')

#Outlier Detection

Q1 = df['MCP'].quantile(0.25)
Q3 = df['MCP'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['MCP'] < Q1 - 1.5 * IQR) | (df['MCP'] > Q3 + 1.5 * IQR)]
print(outliers.shape)

#Data Preprocessing

df.set_index('Date', inplace=True)
df = df.sort_index()                                      

missing_report = df.isna().sum()
print(missing_report)
# Handling missing values (Deleting 6 rows to remove rolling shutter delay NaN values)
df.dropna(subset=['MCP_Rolling'], inplace=True)

#Checking Duplicates & handling
dupes = df.index.duplicated().sum()
if dupes:
    df = df[~df.index.duplicated(keep='first')]

# Extracting Date Values

df['Month']      = df.index.month
df['Quarter']    = df.index.quarter
df['DayOfWeek']  = df.index.dayofweek
df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
df['Day']        = df.index.day

#Lag and rolling Data MCP

df['MCP_lag1'] = df['MCP'].shift(1)
df['MCP_lag7'] = df['MCP'].shift(7)
df['MCP_roll7'] = df['MCP'].rolling(7).mean()
df['MCP_vol30'] = df['MCP'].rolling(30).std()

#Removing NaN values after Lag and rolling 
df.dropna(inplace=True)

#Outlier Handling Windz

from scipy.stats.mstats import winsorize

# Winsorize MCP and store in a new column MCP_Capped
df['MCP_Capped'] = winsorize(df['MCP'], limits=[0.05, 0.05])  # Caps bottom 5% and top 5%


#Noramalising data (Scaling)
scale_cols = [
    'Purchase Bid', 'Sell Bid', 'MCV', 'Final_Volume',
    'MCP_lag1', 'MCP_lag7',
    'MCP_Rolling', 'MCP_roll7',
    'MCP_vol30',
    'MCP_Capped' 
]

from sklearn.preprocessing import MinMaxScaler

# Create the scaler
scaler = MinMaxScaler()

# Fit and transform
df[scale_cols] = scaler.fit_transform(df[scale_cols])

#Saving Scaler for Later Use
import joblib
joblib.dump(scaler, "power_price_scaler.pkl")

# Creating A CSV file for Power BI 
df.to_csv(r"C:/Users/admin/Downloads/PowerBI_ready_dataset.csv")

# XGBoost

# Target variable (already scaled, but keep the original MCP for evaluation)
target_col = 'MCP'
y = df[target_col]

# Select features (you can expand this list)
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


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialize model
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

import xgboost as xgb

# Convert data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

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

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    early_stopping_rounds=10,
    evals=watchlist,
    verbose_eval=False
)

# Predict using the trained model
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

# Evaluation metrics
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

from sklearn.metrics import root_mean_squared_error

print("XGBoost Regression Performance:")
print(f"Test MAE  : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"Test RMSE : {root_mean_squared_error(y_test, y_pred):.3f}")
print(f"Test MAPE : {mape(y_test, y_pred):.2f}%")

#Visualize

plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test.values, label='Actual MCP', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted MCP (XGBoost)', color='green')
plt.title('XGBoost Test Prediction vs Actual MCP')
plt.xlabel('Date')
plt.ylabel('MCP (₹)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


'''Feature importance plot
Understand which features are driving predictions:
'''

from xgboost import plot_importance
plot_importance(model, max_num_features=15, importance_type='gain')
plt.title("Top Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()