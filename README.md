# AI-Powered-Momentum-Based-Algorithmic-Trading-Bot
### Objective:
Develop an AI-driven momentum trading bot that predicts short-term stock price movements using machine learning and executes trades based on signals. 

#### Preliminary: Have the ticker for the stocks or equities handy(ex, AMZN, MSFT, etc)
 
## Collecting and pre-processing of the data:
### a. Install  python libraries

Pandas- for data processing

Yfinance- historical stock prices

Numpy- numerical computations

ta- compute technical indicators

### b.	Fetch Historical data

We'll use yfinance to fetch historical OHLCV (Open, High, Low, Close, Volume) data.

### c.	Clean and Preprocess the data

### d.	 Feature Engineering (Technical Indicators)

To improve predictions, we’ll add Moving Averages, RSI, and MACD using the ta library.

### Technical Indicators:

SMA- Simple moving average- averages closing prices over a certain period of time

Exponential Moving Average(EMA)- similar to SMA but gives weight to recent prices.

### How it’s being used:

When SMA (average closing price in the last 10days) crosses above SMA (50 days average- long trend)- signals uptrend (buy signal)
Opposite is Sell signal. 

RSI(0-100)- Relative Strength Index- measures the momentum of Price movements

RSI>70- overbought (sell signal)- this means the price will drop soon
RSI <30(oversold- buy signal)

MACD- Moving Average Convergence Divergence- show the relationship between two EMAs

(EMA9-) When it crosses the signal line- buy signal. When it crosses below the signal- sell signal

Now that I've gathered the stock data and calculated technical indicators (like SMA, RSI, MACD), it's time to extract meaningful features and prepare my dataset for training a machine learning model. This involves:

1.	**Feature Selection**: technical indicators to use in your model.
2.	**Label Creation**: Create a target variable that will indicate if the stock price will go up or down in the next day (classification).
3.	**Train-Test Split**: Split the data into training and testing sets.
4.	**Scaling the Features**: Normalize the features (important for many machine learning algorithms).

## Step-by-Step: 
### 1. Feature Selection
The features that I am currently working with are the technical indicators that I calculated in the previous step:
•	SMA (10, 50): Moving averages
•	RSI: Relative Strength Index
•	MACD: Moving Average Convergence Divergence

I am keeping this machine learning simple, so will use this features directly. I can further engineer features((e.g., the difference between two SMAs)

### 2. Create a Target Variable (Label)

In stock price prediction, the target variable is usually a binary classification problem, where:
•	1 means the price will go up (price closes higher than the previous day)
•	0 means the price will go down (price closes lower than the previous day)

I will create the target variable by comparing the Close prices of consecutive days.

### 3. Prepare the Data for Machine Learning
Now that I have the features and labels, I need to prepare the data for model training.

### 4. Train a Machine Learning Model

**Step 1**: I will use Random Forest- For regression OR classification- common to use  for stock prediction or algorithmic trading.
**Note:** Random Forest models allow you to examine feature importance. This shows which features have the most influence on predictions.

Other Models that I can use as well are: 

•	Linear Regression (for continuous price prediction)

•	Random Forest (for regression or classification)

•	XGBoost (for better performance with structured data)

•	Logistic Regression (if you're predicting whether a stock will go up or down)

**Step 2:** Prepare Data for Model Training

Before training the model, I need to:

•	Split the data into training and test sets.

•	Separate the features (independent variables) and target (dependent variable).

For this example, let's predict whether the price will go up the next day (target variable), and use the technical indicators as features.

**Prediction Accuracy:** My initial model Prediction was **49%**

**Step 3:** Evaluate for Accuracy and Fine Tune 
I had to finetune the model to make the prediction higher using **Hyperparameter Tuning**

•	Random Forest models have several hyperparameters you can tune, such as:

o	n_estimators (number of trees)

o	max_depth (depth of each tree)

o	min_samples_split (minimum samples required to split an internal node)

o	min_samples_leaf (minimum samples required to be at a leaf node)

**Note**:I could have used XGBoost OR LightGBM- to make the accuracy even way higher! OR even used Log Regression:

**Step 4:** 	Connect to a Paper Trading API using Alpaca- Free API for stock trading

•	Sign up, install the python client and automate

•	Modify the trading python bot to place buy/sell orders when the AI model predicts a favorable trade.

**Step 5:**	Implement Risk Management

•	Stop-Loss: Automatically sell if the trade moves against you.

•	Position Sizing: Limit trade size to prevent overexposure.

**Next feature to work on:** Look into More Advanced Features

•	Sentiment analysis of news articles, tweets, or other text data related to stocks might improve accuracy. There are libraries like VADER or TextBlob that can analyze sentiment and return a score.

•	Consider adding lag features (e.g., the previous day's price) and moving averages of returns to make predictions based on past returns rather than just price indicators.

