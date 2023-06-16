# 📊 Yahoo Finance - TESLA Stock Analysis Guide 🚀
This tutorial will guide you through a simple pipeline to analyze the historical stock prices of TESLA using the Python yfinance library, and visualize data, pre-process it and make forecasting with Machine Learning.

### 📌 Setup
🔗 Helpful Links:

- **Yahoo Finance Guide**
- **Yfinance Python Library**
- **Yahoo Finance Ticker Lookup**

We'll need some important libraries to get started. Run the following to import all necessary packages:

python
Copy code
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.metrics import mean_absolute_error
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

### 📈 Fetching TESLA Data

python
Copy code
data = yf.download ("TSLA")

### 🎨 Visualizing the Data
We will plot the data using Matplotlib and Seaborn:

python
Copy code
def plot_columns(df):
  num_col = len(df.columns)
  fig, axes = plt.subplots(num_col, figsize=(10, 2.5*num_col))
  for i, column in enumerate(df.columns):
    sns.lineplot(ax=axes[i], data=df, x=df.index, y=column)
    axes[0].set_title('TESLA Finance 2010 - 2023')
  plt.tight_layout()
  plt.show()

plot_columns(data)

plt.figure(figsize=(15,5))
sns.lineplot(data['Open'])
plt.title('TESLA FINANCIAL TIME SERIES')

### 💾 Data Pre-processing
1. Split the data into train and test datasets with a split time of 2020-01-01:

python
Copy code
split_time = '2020-01-01'
train_data = data.loc[:split_time]
test_data = data.loc[split_time:]
2. Focus on 'Open' stock data:

python
Copy code
train_open_data = train_data['Open']
valid_open_data = test_data['Open']

# SHOW SERIES
print(f'Train the Open Data \n{train_open_data.head()}\n')
print(f'Validation for the Open Data \n{valid_open_data.head()}\n')
Now, we can begin to forecast with Machine Learning.

For a detailed Machine Learning process using tensorflow, follow this notebook and copy the steps: **TESLA Stock Forecasting Notebook**

After training, check your model history insights and evaluate the model.

## 📊 Outliers Detection using Boxplot
Outliers are data points that are significantly different from the rest of the data. In the context of our analysis, we're focusing on the "Volume" variable, which is suspected to have outliers. Such outliers can distort our analysis, and to prevent this, we're going to remove this variable from our dataset.

# Remove Volume variable from dataset
del data['Volume']

# Plot data after deletion of Volume
sns.boxenplot(data)

This box plot gives us a visual representation of our data distribution, making it easier to spot any remaining outliers. Now we can proceed with a cleaner dataset for a more accurate analysis!

Remember, data cleaning is an essential step in any data analysis task. Always keep an eye out for any potential issues in your data that may skew your results! 👀🧹

# ⚠️ Anomaly Detection Techniques and their Application on Tesla Stock
Anomaly detection is a critical part of data analysis and has applications in many fields. In this article, we explore different anomaly detection techniques and their application on Tesla's stock price data.

First, install **adtk** library by running:

python
Copy code
pip install adtk
Then, import required libraries and modules:

python
Copy code
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
Continue the process for outlier detection, and model evaluation.

## Threshold Detector
Threshold detection is a commonly used technique for detecting anomalies in various applications. A threshold value is set, and any readings that exceed this threshold value are considered anomalous. The success of this technique depends on setting an appropriate threshold value, which can be determined through statistical analysis or by domain experts.

Threshold_detector = ThresholdAD(low=3, high=140)
anomalies = Threshold_detector.detect(open_data)
plot(open_data,anomaly=anomalies, anomaly_color="orange", anomaly_tag='marker')
plt.title("Threshold Anomaly Detection TESLA")

## Quantile Detector
A quantile detector is a statistical tool used to detect the presence of outliers or anomalies in a dataset. It works by dividing the dataset into equal parts based on the distribution of the data and then identifying any data points that fall outside of the expected range.

quantile_detector = QuantileAD(low=0.1, high=0.98)
anomalies = quantile_detector.fit_detect(open_data)
plot(open_data,anomaly=anomalies, anomaly_color="orange", anomaly_tag="marker")
plt.title("Quantile Anomalies Detection")

# Inter Quartile Detector
The Inter Quartile Detector works by analyzing the inter-quartile range of historical data, which is the range between the 25th (1st) and 75th (3rd) percentiles of a dataset. Anomalies are detected if the values fall outside of this range.

iqr_detector = InterQuartileRangeAD(c=1.5)
anomalies = iqr_detector.fit_detect(open_data)
plot(open_data,anomaly=anomalies, anomaly_color="orange", anomaly_tag="marker")
plt.title("Inter Quantile Range Anomaly Detection")

## Persistent Detector
This detector compares time series values with the values of their preceding time windows, and identifies a time point as anomalously large if the change of value from its preceding average or median is significantly different.

persist_detector = PersistAD(c=15, side='positive')
anomalies = persist_detector.fit_detect(open_data)
plot(open_data,anomaly=anomalies, anomaly_color='orange')
plt.title('Positive Persist Anomaly Detector')

persist_detector = PersistAD(c=15, side='negative')
anomalies = persist_detector.fit_detect(open_data)
plot(open_data,anomaly=anomalies, anomaly_color='orange')
plt.title('Negative Persist Anomaly Detector')

## Volatility Detector
Volatility refers to the degree of variation in the values of a data set over time. The Volatility Detector is used to detect shifts in this volatility, which may indicate changes in the underlying factors affecting the data.

volatility_detector = VolatilityShiftAD(c=6.0, side='positive', window=30)
anomalies = volatility_detector.fit_detect(open_data)
plot(open_data, anomaly=anomalies, anomaly_color='orange')
plt.title('Volatility Anomaly Detection')

These various anomaly detection techniques provide different lenses through which to view and understand the Tesla stock price data. Understanding these anomalies can be crucial in making informed decisions and identifying potential risks and opportunities.

# 🎉 Done!
Congrats, you've completed the process of TESLA Stock Analysis! You learned how to fetch historical market data, visualize it, pre-process it, perform time-series forecasting with Machine Learning, and identify any anomalies. Continue to adjust and refine your model as needed.

The world of financial analysis and prediction is yours to conquer! 🚀