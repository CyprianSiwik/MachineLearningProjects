#Project 3 - using yahoo finance data and ml algorithms - 3/1/25

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#Download stock data - ticker is the stock - can change the start and end date
ticker = "AAPL"
df = yf.download(ticker, start="2022-01-01", end = "2024-01-01")

#Feature Engineering

#Create moving averages
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()

#Daily returns
df['Return'] = df['Close'].pct_change()

#Momentum (Difference between today's close and 3 days ago)
df['Momentum'] = df['Close'] - df['Close'].shift(3)

#Exponential moving averages (EMA)
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

#RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

sns.set(style="whitegrid")

#Bollinger Bands
# Bollinger Bands
rolling_std = df[['Close']].rolling(window=5).std().iloc[:, 0]  # Ensure we get a Series

# Check what rolling_std is to make sure it's a Series
print(f"rolling_std: {rolling_std.head()}")  # Inspect rolling standard deviation

# If rolling_std is a Series, we can safely assign the values
df['Upper_BB'] = df['MA5'] + (2 * rolling_std)
df['Lower_BB'] = df['MA5'] - (2 * rolling_std)

# Print the first few rows to verify
print(df[['Upper_BB', 'Lower_BB']].head())


#df['Upper_BB'] = df['MA5'] + (2 * df['Close'].rolling(window=5).std())
#df['Lower_BB'] = df['MA5'] - (2 * df['Close'].rolling(window=5).std())

#Average True Range (ATR)
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()

#Target Variable: 1 if price goes up next day, 0 if down
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

#Drop NaN values
df.dropna(inplace=True)

#Display the new dataset with features
print(df[['Close', 'MA5', 'MA10', 'Return', 'Momentum', 'EMA12', 'EMA26', 'RSI', 'Upper_BB', 'Lower_BB', 'ATR', 'Target']].head())

#Prepare Data for Training
features = ['MA5', 'MA10', 'Return', 'Momentum', 'EMA12', 'EMA26', 'RSI', 'Upper_BB', 'Lower_BB', 'ATR']
x = df[features]
y = df['Target']

#Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Train the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

#Show classification report
print(classification_report(y_test, y_pred))

#Feature Importance Plot
feature_importance = pd.Series(model.feature_importances_, index=features)
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importance, y=feature_importance.index, palette='Blues', width=0.55, alpha=0.85)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.gca().spines['top'].set_linewidth(1)  # Set top border thickness
plt.gca().spines['bottom'].set_linewidth(1)  # Set bottom border thickness
plt.gca().spines['left'].set_linewidth(1)  # Set left border thickness
plt.gca().spines['right'].set_linewidth(1)  # Set right border thickness

#plt.gca().set_facecolor('lightgrey')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance for Stock Movement Prediction")
plt.show()

#Visualizing Predictions vs Actual Stock Prices
dates = df.index[-len(y_test):]
plt.figure(figsize=(12,6))
plt.plot(dates, y_test, label="Actual", color='blue', alpha=0.6)
plt.plot(dates, y_pred, label="Predicted", color='red', linestyle='dashed')
plt.legend()
plt.title(f"{ticker} Stock Movement Prediction: Actual vs Predicted")
plt.show()





