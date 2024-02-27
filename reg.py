from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math, random

def LIN_REG_ALGO(df,forecast_days=15):
    #No of days to be forcasted in future
    forecast_out = int(forecast_days)
    #Price after n days
    original_highcharts = df[['Date','Close']]
    original_highcharts['Close'] = original_highcharts['Close'].round(2)
    original_highcharts1=original_highcharts.values.tolist()
    original_highcharts=original_highcharts.values.tolist()
    print("____Regression_________")
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    end_dates = df['Date'].max()
    #New df with only relevant data
    df_new=df[['Close','Close after n days']]
    #Structure data for train, test & forecast
    #lables of known data, discard last 35 rows
    y =np.array(df_new.iloc[:-forecast_out,-1])
    y=np.reshape(y, (-1,1))
    #all cols of known data except lables, discard last 35 rows
    X=np.array(df_new.iloc[:-forecast_out,0:-1])
    #Unknown, X to be forecasted
    X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
    #Traning, testing to plot graphs, check accuracy
    X_train=X[0:int(0.8*len(df)),:]
    X_test=X[int(0.8*len(df)):,:]
    y_train=y[0:int(0.8*len(df)),:]
    y_test=y[int(0.8*len(df)):,:]
    # Feature Scaling===Normalization
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted=sc.transform(X_to_be_forecasted)
    #Training

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    #Testing
    y_test_pred=clf.predict(X_test)
    y_test_pred=y_test_pred*(1.04)
    error_lr = round(math.sqrt(mean_squared_error(y_test, y_test_pred)),2)
    r2_error_reg = round(r2_score(y_test, y_test_pred),2)

    #Forecasting
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set=forecast_set*(1.04)
    mean=forecast_set.mean()
    lr_pred=forecast_set[0,0]
    sorted_test= original_highcharts1[-len(y_test_pred):]
    for i in range(0,len(sorted_test)):
        sorted_test[i][1]=round(y_test_pred[i][0],2)
    last_date = datetime.strptime(end_dates, '%Y-%m-%d')
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 16)]  # Forecasting for the next 15 days
    lastfiftendayforecast = [[date.strftime('%Y-%m-%d'), round(value[0],2)] for date, value in zip(forecast_dates, forecast_set)]
    change = lastfiftendayforecast[-1][1] - original_highcharts[-1][1]
    percentage_change = round(((change / original_highcharts[-1][1]) * 100),2)
    return {'reg_pred':lastfiftendayforecast,'error_reg':error_lr,
                'r2_error_reg':r2_error_reg,'original_highcharts':original_highcharts,'test':sorted_test,'percentage_change':percentage_change}
    