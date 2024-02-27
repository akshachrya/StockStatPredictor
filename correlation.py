from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np



def correlation_finder(df1,df2,time,quote,key):
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    # Merge DataFrames on the 'Date' column
    merged_df = pd.merge(df1, df2, on='Date')
    merged_df = merged_df.rename(columns={'Close_x': quote, 'Close_y': key})
    end_date = merged_df['Date'].max()
    if time=='Overall':
        start_date = merged_df['Date'].min()
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    if time=='7day':
        start_date = end_date - pd.DateOffset(days=7)
        merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    if time=='15day':
        start_date = end_date - pd.DateOffset(days=15)
        merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    if time=='1month':
        start_date = end_date - pd.DateOffset(days=30)
        merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    if time=='3month':
        start_date = end_date - pd.DateOffset(days=90)
        merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    if time=='6month':
        start_date = end_date - pd.DateOffset(days=180)
        merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    if time=='1year':
        start_date = end_date - pd.DateOffset(days=365)
        merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
        correlation_matrix = merged_df.corr()
        coeff=round(correlation_matrix[quote][key],2)
    # Calculate day duration
    day_duration = (end_date - start_date).days
    return coeff

def datamaker(data,quote):
    main=[]
    labels=['Overall','7day','15day','1month','3month','6month','1year']
    for key, value in data.items():
        d={}
        if len(value)!=0 and quote != key:
            overalldata = data[quote]
            seconddata=data[key]
            a=[]
            keys=[]
            for i in labels:
                coeffs=a.append([i,correlation_finder(overalldata,seconddata,i,quote,key)])
                d[key]=a
            main.append(d)
    return main

def finalcorr(data):
    flat_data = [(key, val) for dct in data for key, val in dct.items()]

    # Create a DataFrame
    df = pd.DataFrame(flat_data, columns=['Company', 'Values'])
    # Expand the 'Values' column into multiple columns
    df[['Overall', '7day', '15day', '1month', '3month', '6month', '1year']] = pd.DataFrame(df['Values'].tolist(), index=df.index)

    # Drop the original 'Values' column
    df.drop(columns=['Values'], inplace=True)

    for col in ['Overall', '7day', '15day', '1month', '3month', '6month', '1year']:
        df[col] = df[col].apply(lambda x: float(x[1]))

    # Top five positive correlations for each time period
    positive_correlations = {}
    for col in df.columns[1:]:
        positive_correlations[col] = df.nlargest(5, col)

    # Top five negative correlations for each time period
    negative_correlations = {}
    for col in df.columns[1:]:
        negative_correlations[col] = df.nsmallest(5, col)

    # Neutral correlations
    neutral_correlations = {}
    for col in df.columns[1:]:
        neutral_correlations[col] = df[
            (df[col] >= -0.3) & (df[col] <= 0.3)
        ].nsmallest(5, col)
    overall_pos={}
    overall_neg={}
    overall_neu={}
    for col, data in positive_correlations.items():
        overall_pos[col]=data[['Company',col]].to_dict('records')
    for col, data in negative_correlations.items():
        overall_neg[col]=data[['Company',col]].to_dict('records')
    for col, data in neutral_correlations.items():
        overall_neu[col]=data[['Company',col]].to_dict('records')
    return {'pos':overall_pos,'neg':overall_neg,'neutral':overall_neu}