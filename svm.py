import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from datetime import datetime,timedelta
import math, random

def SVM_ALGO(df,forecast_days=15):
    df =df[['Date','Close']]
    original_highcharts = df[['Date','Close']]
    original_highcharts['Close'] = original_highcharts['Close'].round(2)
    original_highcharts1=original_highcharts.values.tolist()
    original_highcharts=original_highcharts.values.tolist()
    print("____SVM_________")
    end_dates = df['Date'].max()
    # Splitting the data into features (X) and target variable (y)
    # Here, we are using a simple autoregressive model
    X = df['Close'].shift(1).fillna(0)  # Features (using lagged values)
    y = df['Close']                      # Target variable

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshaping the data to fit SVR input requirements
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    y_train = y_train.values
    y_test = y_test.values
    # Create and train the Support Vector Regression model
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_model.fit(X_train, y_train)

    y_pred_test = svr_model.predict(X_test)
    sorted_test= original_highcharts1[-len(y_pred_test):]
    for i in range(0,len(sorted_test)):
        sorted_test[i][1]=round(y_pred_test[i],2)
    # Evaluating the model
    error_svm = round(math.sqrt(mean_squared_error(y_test, y_pred_test)),2)
    r2_error_svm = round(r2_score(y_test, y_pred_test),2)
    # Forecasting for the next 15 days
    # For forecasting, we need to use the last known value as the input
    last_known_value = df['Close'].iloc[-1]
    forecasted_values = []

    for i in range(forecast_days):
        # Using the last known value to predict the next value
        next_value = svr_model.predict([[last_known_value]])
        forecasted_values.append(next_value[0])
        last_known_value = next_value[0]
    last_date = datetime.strptime(end_dates, '%Y-%m-%d')
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]  # Forecasting for the next 15 days
    lastfiftendayforecast = [[date.strftime('%Y-%m-%d'), round(value,2)] for date, value in zip(forecast_dates, forecasted_values)]
    change = lastfiftendayforecast[-1][1] - original_highcharts[-1][1]
    percentage_change = round(((change / original_highcharts[-1][1]) * 100),2)
    return {'svm_pred':lastfiftendayforecast,'error_svm':error_svm,
                'r2_error_svm':r2_error_svm,'original_highcharts':original_highcharts,'test':sorted_test,'percentage_change':percentage_change}

