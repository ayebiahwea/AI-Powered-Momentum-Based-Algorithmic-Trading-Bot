
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# List of stock tickers
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "MYND", "INTC","TCEHY", "LUNR", "AMZN", "NFLX"]

# Define date range
start_date = "2023-01-01"
end_date = "2024-01-01"

# Dictionary to store data for each stock
stock_data = {}

for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    if data.empty:
        print(f"No data found for {ticker}. Skipping...")
        continue
    
    # Fix: Use ffill() instead of deprecated fillna()
    data.ffill(inplace=True)

    # Compute Technical Indicators manually
    sma_10 = data["Close"].rolling(window=10).mean()
    sma_50 = data["Close"].rolling(window=50).mean()
    rsi = 100 - (100 / (1 + data["Close"].diff().rolling(window=14).apply(lambda x: (x[x > 0].sum() / abs(x[x < 0].sum())), raw=False)))
    macd = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()

    # Ensure the indicator values are 1D and assign to DataFrame columns
    data["SMA_10"] = sma_10.squeeze()  # squeeze to ensure 1D
    data["SMA_50"] = sma_50.squeeze()  # squeeze to ensure 1D
    data["RSI"] = rsi.squeeze()        # squeeze to ensure 1D
    data["MACD"] = macd.squeeze()      # squeeze to ensure 1D

    # Create the target variable (1 for up, 0 for down)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # Drop NaN values
    data.dropna(inplace=True)
    
    # Store in dictionary
    stock_data[ticker] = data

    # Select features (exclude 'Close' and 'Target')
    features = data[["SMA_10", "SMA_50", "RSI", "MACD"]]

    # Prepare the data for machine learning
    # Define features (X) and target (y)
    X = features
    y = data["Target"]
    
    # Split the data into training and testing sets (80% train, 20% test)
    if len(X) < 2:
        print(f"Not enough data for {ticker}. Skipping...")
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Standardize the features (important for many models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optionally, store scaled datasets as separate DataFrames
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save scaled datasets to CSV files
    X_train_df.to_csv(f"{ticker}_X_train.csv", index=False)
    X_test_df.to_csv(f"{ticker}_X_test.csv", index=False)
    
    print(f"Data for {ticker} preprocessed and saved successfully!\n")

# Optionally, merge all stock data into one CSV file if there is any valid stock data
if stock_data:
    merged_data = pd.concat(stock_data.values(), keys=stock_data.keys(), names=["Ticker", "Date"])
    merged_data.to_csv("all_stocks_data_merged.csv")
else:
    print("No valid stock data to merge.")

print("All stock data saved successfully!")

# Assuming 'data' contains all features and target, 'target' is 1 if the price goes up, else 0
if not data.empty:
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price goes up, else 0

# Features are all technical indicators
features = ['SMA_10', 'SMA_50', 'RSI', 'MACD']
X = data[features]

# Target is whether the price goes up or down
y = data['target']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# Get the best model from grid search
best_model = grid_search.best_estimator_

importances = model.feature_importances_
feature_names = list(X.columns)  # Ensure feature names are a list

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).reset_index(drop=True)  # Reset index to avoid MultiIndex issues

# Convert Feature column to string
importance_df['Feature'] = importance_df['Feature'].astype(str)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()



import alpaca_trade_api as tradeapi

API_KEY = "PK3FKPSDU0D6KNVND9LQ"
API_SECRET ="1ztmDpTHcIypURlpa50jEKtYpQ9h6hqztiHcGam1"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

account = api.get_account()
print(f"Account Status: {account.status}")



def place_orders(symbols, qty, side):
    for symbol in symbols:
        try:
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            print(f"Order placed: {side} {qty} of {symbol}")
        except Exception as e:
            print(f"Order failed for {symbol}: {e}")

# Example: Buy 1 share each of AAPL, MSFT, TSLA, MYND, INTC,TCEHY, LUNR, AMZN, NFLX
tickers = ["AAPL", "MSFT", "TSLA","MYND", "INTC","TCEHY", "LUNR", "AMZN", "NFLX"]
place_orders(tickers, 1, "buy")


