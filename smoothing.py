***************(All Smoothing Techniques)*********************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

# Sample time series data
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140]

df = pd.DataFrame(data, columns=['Passengers'])
df['Month'] = pd.date_range(start='2020-01', periods=len(df), freq='M')
df.set_index('Month', inplace=True)

# --- 1. Simple Moving Average (window=3)
df['SMA_3'] = df['Passengers'].rolling(window=3).mean()

# --- 2. Weighted Moving Average (manual)
weights = np.array([0.1, 0.3, 0.6])  # More weight to recent
def weighted_moving_average(series, weights):
    return np.convolve(series, weights[::-1], mode='valid')

df['WMA_3'] = pd.Series(weighted_moving_average(df['Passengers'], weights), index=df.index[2:])

# --- 3. Exponential Smoothing (Single)
model_ses = SimpleExpSmoothing(df['Passengers']).fit(smoothing_level=0.2, optimized=False)
df['SES'] = model_ses.fittedvalues

# --- 4. Double Exponential Smoothing (Holt’s Linear Trend)
model_holt = Holt(df['Passengers']).fit(smoothing_level=0.8, smoothing_trend=0.2)
df['Holt'] = model_holt.fittedvalues

# --- 5. Triple Exponential Smoothing (Holt-Winters Seasonal)
model_hw = ExponentialSmoothing(df['Passengers'], seasonal='add', seasonal_periods=12).fit()
df['Holt_Winters'] = model_hw.fittedvalues

# --- Plot all
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Original', linewidth=2)
plt.plot(df['SMA_3'], label='Simple Moving Avg')
plt.plot(df['WMA_3'], label='Weighted Moving Avg')
plt.plot(df['SES'], label='Exponential Smoothing')
plt.plot(df['Holt'], label='Double Exp Smoothing')
plt.plot(df['Holt_Winters'], label='Holt-Winters')
plt.legend()
plt.title("Smoothing Techniques in Time Series")
plt.grid()
plt.show()






//simpple exponential smothinh

import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Load data
sales_data = pd.read_csv('/content/Simple_Exponential_smoothing_dataset.csv')
sales = sales_data["Sales"]

# Fit model with given alpha
model = SimpleExpSmoothing(sales).fit(smoothing_level=0.3, optimized=False)

# Get smoothed values and forecast
sales_data["Smoothed"] = model.fittedvalues
forecast_nov = model.forecast(1)[0]

# Output
print(sales_data)
print(f"\nForecasted Sales for November: {forecast_nov:.2f}")


//double exponential

import pandas as pd
from statsmodels.tsa.holtwinters import Holt

# Load data
df = pd.read_csv('/content/Double_exponential_smoothing_dataset.csv')
sales = df["Sales"]

# Fit Holt's model with alpha and beta
model = Holt(sales).fit(smoothing_level=0.3, smoothing_trend=0.2, optimized=False)

# Add smoothed and trend values
df["Smoothed"] = model.fittedvalues
df["Trend"] = model.level + model.trend - model.fittedvalues  # Optional: estimate trend

# Forecast for Nov and Dec
forecast = model.forecast(2)
forecast_data = pd.DataFrame({
    "Month": ["Nov", "Dec"],
    "Sales": [None, None],
    "Smoothed": forecast,
    "Trend": [model.trend[-1]] * 2
})

df = pd.concat([df, forecast_data], ignore_index=True)

# Output
print(df)
print(f"\nForecasted Sales for November: {forecast.iloc[0]:.2f}")
print(f"Forecasted Sales for December: {forecast.iloc[1]:.2f}")


//triple

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data
df = pd.read_csv('/content/Triple_exponential_smoothing_dataset.csv')
sales = df["Sales"]

# Fit the model with additive or multiplicative seasonality
model_add = ExponentialSmoothing(
    sales,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit(smoothing_level=0.3, smoothing_slope=0.2, smoothing_seasonal=0.1, optimized=False)

# Forecast 3 months ahead (e.g., for December)
forecast_add = model_add.forecast(3)

# Output
print("Additive Model Forecast for next 3 periods:")
print(forecast_add)

