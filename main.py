import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Fetch stock data
stock_data = yf.download('TSLA', start='2020-01-01', end='2025-01-01')  # Replace 'AAPL' with any stock symbol
stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(stock_data.head())


# Feature engineering
stock_data['Moving_Avg_15'] = stock_data['Close'].rolling(window=5).mean()
stock_data['Moving_Avg_30'] = stock_data['Close'].rolling(window=10).mean()

# Target variable: Predict the next day's closing price
stock_data['Target'] = stock_data['Close'].shift(-1)

# Drop rows with NaN values (due to rolling averages or shifting)
stock_data = stock_data.dropna()

# Features and target
features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Moving_Avg_15', 'Moving_Avg_30']]
target = stock_data['Target']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)
print('Predictions:', y_pred)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")




# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices', alpha=0.7)
plt.plot(y_pred, label='Predicted Prices', alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Data Points")
plt.ylabel("Stock Price")
plt.show()
