import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Not used, can be removed if desired
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


# Set seaborn style for professional visuals
sns.set(style="whitegrid")

#############################
# 1. Data Acquisition
#############################
assets = ["NVDA", "MSFT", "GOOGL", "AMZN", "ARKQ", "BOTZ", "QQQ", "SPY", "TLT", "GLD"]

print("Downloading data...")
data = yf.download(assets, start="2019-01-01", end="2024-01-01", group_by="ticker")

# Debug: Print data structure and column names
print("\nData structure preview:")
print(data.head())
print("\nColumn names:")
print(data.columns)

# Extract Adjusted Close prices. If "Adj Close" is missing, fall back to "Close".
try:
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(1):
            price_data = data.xs("Adj Close", axis=1, level=1)
            print("Extracted 'Adj Close' successfully (MultiIndex).")
        elif "Close" in data.columns.get_level_values(1):
            price_data = data.xs("Close", axis=1, level=1)
            print("Warning: 'Adj Close' not found. Using 'Close' prices instead.")
        else:
            raise KeyError("Error: Neither 'Adj Close' nor 'Close' found in MultiIndex columns.")
    elif "Adj Close" in data.columns:
        price_data = data["Adj Close"]
        print("Extracted 'Adj Close' successfully (Single-level DataFrame).")
    elif "Close" in data.columns:
        price_data = data["Close"]
        print("Warning: 'Adj Close' not found in single-level DataFrame. Using 'Close' prices instead.")
    else:
        raise KeyError("Error: Neither 'Adj Close' nor 'Close' found in DataFrame.")
except KeyError as e:
    print(e)
    print("\nData structure:")
    print(data.head())
    exit()

# Forward-fill missing values if any
price_data.fillna(method="ffill", inplace=True)
print("\nFinal Price Data:")
print(price_data.head())

#############################
# 2. Rolling Portfolio Optimization
#############################
# Calculate daily returns and set up rolling window parameters
returns = price_data.pct_change().dropna()
rolling_window = 252  # One-year rolling window
rolling_mean_returns = returns.rolling(window=rolling_window).mean()
rolling_cov_matrix = returns.rolling(window=rolling_window).cov()

# Use the most recent window for optimization
current_mean_returns = rolling_mean_returns.iloc[-1]
current_cov_matrix = returns.cov()  # Using overall covariance for simplicity

def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = np.sum(mean_returns * weights) * 252
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (annual_return - 0.02) / annual_volatility  # Assuming 2% risk-free rate
    return -sharpe_ratio  # Negative for minimization

def optimize_portfolio(mean_returns, cov_matrix):
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 0.4) for _ in assets)  # Maximum allocation 40% per asset
    initial_weights = np.array([1 / len(assets)] * len(assets))
    opt_result = sco.minimize(
        portfolio_performance, initial_weights, args=(mean_returns, cov_matrix),
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    return opt_result.x

optimal_weights = optimize_portfolio(current_mean_returns, current_cov_matrix)
print("\nInitial Optimized Portfolio Weights:")
print(dict(zip(assets, optimal_weights)))

#############################
# 3. AI Market Prediction Using Multi-Stock LSTM
#############################
def create_lstm_model(input_shape, output_dim):
    """Builds an LSTM model for multi-stock prediction."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(output_dim)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# Scale the entire price data for multiple assets
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(price_data)

# Create a multi-stock dataset with a look_back window
X, Y = [], []
look_back = 60
for i in range(len(scaled_data) - look_back - 1):
    X.append(scaled_data[i:(i + look_back)])
    Y.append(scaled_data[i + look_back])
X, Y = np.array(X), np.array(Y)

# Create and train the LSTM model
lstm_model = create_lstm_model((look_back, len(assets)), len(assets))
lstm_model.fit(X, Y, epochs=10, batch_size=16, verbose=1)

# Predict future prices for the latest available window
predicted_prices = lstm_model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Calculate predicted returns for each asset using the last prediction vs. current prices
predicted_returns = (predicted_prices[-1] - price_data.iloc[-1].values) / price_data.iloc[-1].values
# Identify top 3 predicted performers and increase their weights by 10%
top_performers = np.argsort(predicted_returns)[-3:]
print("\nTop 3 Predicted Performers Indices:", top_performers)
optimal_weights[top_performers] *= 1.1

#############################
# 4. Volatility-Based Risk Adjustment
#############################
# Calculate the 30-day rolling volatility for key AI stocks
ai_keys = ["NVDA", "MSFT", "GOOGL", "AMZN"]
ai_volatility = returns[ai_keys].rolling(window=30).std().iloc[-1].mean()

if ai_volatility > 0.04:
    print("High volatility detected! Adjusting portfolio to increase hedge assets (bonds & gold).")
    optimal_weights[:4] *= 0.9  # Reduce allocation for key AI stocks
    optimal_weights[-2:] += 0.1  # Increase allocation for TLT (bonds) & GLD (gold)

#############################
# 5. Monte Carlo Simulation (Including Black Swan Events)
#############################
n_simulations = 10000
simulated_returns = np.zeros(n_simulations)

for i in range(n_simulations):
    rand_weights = np.random.dirichlet(np.ones(len(assets)))
    sim_return = np.sum(rand_weights * current_mean_returns) * 252
    # Introduce a 1% chance per year for a market crash (40% loss)
    if np.random.rand() < 0.01:
        sim_return *= 0.6
    simulated_returns[i] = sim_return

plt.figure(figsize=(10, 6))
sns.histplot(simulated_returns, bins=50, kde=True, color="blue", alpha=0.6)
plt.axvline(x=np.percentile(simulated_returns, 5), color="red", linestyle="--", label="5% Worst Case")
plt.axvline(x=np.percentile(simulated_returns, 95), color="green", linestyle="--", label="95% Best Case")
plt.legend()
plt.title("Monte Carlo Simulation with Black Swan Events", fontsize=14)
plt.xlabel("Annualized Return")
plt.ylabel("Frequency")
plt.show()

#############################
# 6. Final Portfolio Visualization
#############################
portfolio_df = pd.DataFrame({"Asset": assets, "Allocation (%)": optimal_weights * 100})
portfolio_df["Allocation (%)"] = portfolio_df["Allocation (%)"].round(2)

plt.figure(figsize=(8, 6))
colors = sns.color_palette("pastel")
plt.pie(portfolio_df["Allocation (%)"], labels=portfolio_df["Asset"], autopct="%1.1f%%", colors=colors, startangle=140)
plt.title("Optimized AI Investment Portfolio", fontsize=14)
plt.show()

