from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime,timedelta
import pandas as pd
import numpy as np

import math, random
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
#******************** ARIMA SECTION ********************
def ARIMA_ALGO(df,forecast_days=15):
    df =df[['Date','Close']]
    original_highcharts = df[['Date','Close']]
    original_highcharts['Close'] = original_highcharts['Close'].round(2)
    original_highcharts1=original_highcharts.values.tolist()
    original_highcharts=original_highcharts.values.tolist()
    print("____ARIMA_________")
    train_data, test_data = train_test_split(df['Close'], test_size=0.2, shuffle=False)
    # Converting 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Training ARIMA model
    history = [x for x in train_data]
    predictions = []
    test_data=test_data.tolist()
    for t in range(len(test_data)):
        model = ARIMA(history, order=(6,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[t]
        history.append(obs)

    # Calculating error
    error_arima = round(math.sqrt(mean_squared_error(test_data, predictions)),2)
    r2_error_arima = round(r2_score(test_data, predictions),2)
    # Forecasting the next 15 days
    forecast_period = forecast_days
    forecast = model_fit.forecast(steps=forecast_period)

    # Creating dates for the forecast period
    last_date = df['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_period + 1).strftime('%Y-%m-%d')[1:]
    lastfiftendayforecast = [[date, round(value,2)] for date, value in zip(forecast_dates, forecast)]
    sorted_test= original_highcharts1[-len(predictions):]
    for i in range(0,len(sorted_test)):
        sorted_test[i][1]=round(predictions[i],2)
    change = lastfiftendayforecast[-1][1] - original_highcharts[-1][1]
    percentage_change = round(((change / original_highcharts[-1][1]) * 100),2)
    return {'arima_pred':lastfiftendayforecast,'error_arima':error_arima,
   'r2_error_arima':r2_error_arima,'original_highcharts':original_highcharts,'test':sorted_test,'percentage_change':percentage_change}