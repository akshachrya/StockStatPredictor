from flask import Flask, render_template, url_for, redirect, flash, request, jsonify
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import math, random
from datetime import datetime
import yfinance as yf
import re
from textblob import TextBlob
from correlation import *
from arima import *
from reg import *
from svm import *

app = Flask(__name__)
app.secret_key = "super secret key"
COMPANY=pd.read_csv('data/ticker_details/Yahoo-Finance-Ticker-Symbols.csv')
# ticker_details=['AMZN','AAPL','GOOG']
forecast_days=15
ticker_details=COMPANY['Ticker'].tolist()

def get_historical_multiple(quote):
    end = datetime.now()
    start = datetime(end.year-5,end.month,end.day)
    main={}
    for i in quote:
        data = yf.download(i, start=start, end=end, ignore_tz=True)
        data=data.iloc[::-1]
        data=data.reset_index()
        data=data[['Date','Close']]
        data=data.to_dict('records')
        main[i]=data
    return main

CORR_DATA=get_historical_multiple(ticker_details)

def get_historical(quote):
	end = datetime.now()
	start = datetime(end.year-5,end.month,end.day)
	data = yf.download(quote, start=start, end=end, ignore_tz=True)
	df = pd.DataFrame(data=data)
	df.to_csv('data/individual/'+quote+'.csv')
	if(df.empty):
	    ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
	    data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
	    #Format df
	    #Last 2 yrs rows => 502, in ascending order => ::-1
	    data=data.iloc[::-1]
	    data=data.reset_index()
	    #Keep Required cols only
	    df=pd.DataFrame()
	    df['Date']=data['date']
	    df['Open']=data['1. open']
	    df['High']=data['2. high']
	    df['Low']=data['3. low']
	    df['Close']=data['4. close']
	    df['Adj Close']=data['5. adjusted close']
	    df['Volume']=data['6. volume']
	    df.to_csv('data/individual/'+quote+'.csv',index=False)
	return df

def calculate_beta(stock_symbol, market_symbol, start_date, end_date):
    # Download historical stock prices
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)['Adj Close']
    market_data = yf.download(market_symbol, start=start_date, end=end_date)['Adj Close']

    # Combine stock and market data into a DataFrame
    data = pd.concat([stock_data, market_data], axis=1)
    data.columns = [stock_symbol, market_symbol]

    # Calculate returns
    returns = data.pct_change().dropna()

    # Calculate covariance matrix
    covariance_matrix = np.cov(returns[stock_symbol], returns[market_symbol])

    # Calculate beta
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta


@app.route("/", methods = ["GET"])
@app.route("/home", methods = ["GET"])
def home():
    company=COMPANY
    company_dict=company.to_dict('records')
    return render_template("frontpage.html",COMPANY=company_dict)

@app.route("/<symbol>")
def viz(symbol):
    # import the data
    quote = symbol.upper()
    end = datetime.today()
    start = end - timedelta(days = 30)
    data=get_historical(quote)
    df=data
    stock = yf.Ticker(quote)
    # Get 52-week high and 52-week low
    info = stock.info
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    fifty_two_week_low = info.get("fiftyTwoWeekLow")
    mkt_cap = info.get("marketCap")
    pe_ratio = info.get("forwardPE")
    div_yield = info.get("dividendYield")
    today_stock=df.iloc[-1:]
    yesterday_stock=df.iloc[-2:]
    date_index=today_stock.index
    date_str = date_index[0].strftime('%Y-%m-%d')
    today_stock=today_stock.to_dict('records')[0]
    today_stock['Date']=date_str
    today_stock['fiftyTwoWeekHigh']=fifty_two_week_high
    today_stock['fiftyTwoWeekLow']=fifty_two_week_low
    today_stock['marketCap']=mkt_cap
    today_stock['forwardPE']=pe_ratio
    today_stock['dividendYield']=div_yield
    yesterday_stock=yesterday_stock.to_dict('records')[0]
    today_stock = {key: round(value, 2) if isinstance(value, float) else value for key, value in today_stock.items()}
    yesterday_stock = {key: round(value, 2) if isinstance(value, float) else value for key, value in yesterday_stock.items()}
    today_stock['openratio']=round(today_stock['Open']-yesterday_stock['Open'],2)
    today_stock['openratioper']=round(((today_stock['Open']-yesterday_stock['Open'])/yesterday_stock['Open'])*100,2)
    today_stock['highratio']=round(today_stock['High']-yesterday_stock['High'],2)
    today_stock['highratioper']=round(((today_stock['High']-yesterday_stock['High'])/yesterday_stock['High'])*100,2)
    today_stock['lowratio']=round(today_stock['Low']-yesterday_stock['Low'],2)
    today_stock['lowratioper']=round(((today_stock['Low']-yesterday_stock['Low'])/yesterday_stock['Low'])*100,2)
    today_stock['closeratio']=round(today_stock['Close']-yesterday_stock['Close'],2)
    today_stock['closeratioper']=round(((today_stock['Close']-yesterday_stock['Close'])/yesterday_stock['Close'])*100,2)
    today_stock['adjcloseratio']=round(today_stock['Adj Close']-yesterday_stock['Adj Close'],2)
    today_stock['adjcloseratioper']=round(((today_stock['Adj Close']-yesterday_stock['Adj Close'])/yesterday_stock['Adj Close'])*100,2)
    today_stock['volumeratio']=round(today_stock['Volume']-yesterday_stock['Volume'],2)
    today_stock['volumeratioper']=round(((today_stock['Volume']-yesterday_stock['Volume'])/yesterday_stock['Volume'])*100,2)
    company=COMPANY
    companymain=company.to_dict('records')
    company_df = company[company['Ticker'] == quote]
    company_df_dict=company_df.to_dict('records')[0]
    return render_template("viz.html",today_stock=today_stock,company_dict=company_df_dict,COMPANY=companymain)


@app.route("/correlation",methods=['GET'])
def correlation():
    symbol = request.args.get('ticker')
    data=CORR_DATA
    corr_data_overall=datamaker(data,symbol)
    final=finalcorr(corr_data_overall)
    return jsonify(final)

@app.route("/predict",methods=['GET'])
def predict():
    # import the data
    symbol = request.args.get('symbol')
    quote = symbol.upper()
    end = datetime.today()
    start = end - timedelta(days = 30)
    data=get_historical(quote)
    df=data
    stock = yf.Ticker(quote)
    # Get 52-week high and 52-week low
    info = stock.info
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    fifty_two_week_low = info.get("fiftyTwoWeekLow")
    mkt_cap = info.get("marketCap")
    pe_ratio = info.get("forwardPE")
    div_yield = info.get("dividendYield")
    today_stock=df.iloc[-1:]
    yesterday_stock=df.iloc[-2:]
    date_index=today_stock.index
    date_str = date_index[0].strftime('%Y-%m-%d')
    today_stock=today_stock.to_dict('records')[0]
    today_stock['Date']=date_str
    today_stock['fiftyTwoWeekHigh']=fifty_two_week_high
    today_stock['fiftyTwoWeekLow']=fifty_two_week_low
    today_stock['marketCap']=mkt_cap
    today_stock['forwardPE']=pe_ratio
    today_stock['dividendYield']=div_yield
    yesterday_stock=yesterday_stock.to_dict('records')[0]
    today_stock = {key: round(value, 2) if isinstance(value, float) else value for key, value in today_stock.items()}
    yesterday_stock = {key: round(value, 2) if isinstance(value, float) else value for key, value in yesterday_stock.items()}
    today_stock['openratio']=round(today_stock['Open']-yesterday_stock['Open'],2)
    today_stock['openratioper']=round(((today_stock['Open']-yesterday_stock['Open'])/yesterday_stock['Open'])*100,2)
    today_stock['highratio']=round(today_stock['High']-yesterday_stock['High'],2)
    today_stock['highratioper']=round(((today_stock['High']-yesterday_stock['High'])/yesterday_stock['High'])*100,2)
    today_stock['lowratio']=round(today_stock['Low']-yesterday_stock['Low'],2)
    today_stock['lowratioper']=round(((today_stock['Low']-yesterday_stock['Low'])/yesterday_stock['Low'])*100,2)
    today_stock['closeratio']=round(today_stock['Close']-yesterday_stock['Close'],2)
    today_stock['closeratioper']=round(((today_stock['Close']-yesterday_stock['Close'])/yesterday_stock['Close'])*100,2)
    today_stock['adjcloseratio']=round(today_stock['Adj Close']-yesterday_stock['Adj Close'],2)
    today_stock['adjcloseratioper']=round(((today_stock['Adj Close']-yesterday_stock['Adj Close'])/yesterday_stock['Adj Close'])*100,2)
    today_stock['volumeratio']=round(today_stock['Volume']-yesterday_stock['Volume'],2)
    today_stock['volumeratioper']=round(((today_stock['Volume']-yesterday_stock['Volume'])/yesterday_stock['Volume'])*100,2)
    company=COMPANY
    companymain=company.to_dict('records')
    company_df = company[company['Ticker'] == quote]
    company_df_dict=company_df.to_dict('records')[0]
    return render_template("predict.html",today_stock=today_stock,company_dict=company_df_dict,COMPANY=companymain)

@app.route("/stat",methods=['GET'])
def statistics():
    symbol = request.args.get('symbol')
    quote = symbol.upper()
    end = datetime.today()
    start = end - timedelta(days = 30)
    data=get_historical(quote)
    df=data
    stock = yf.Ticker(quote)
    # Get 52-week high and 52-week low
    info = stock.info
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    fifty_two_week_low = info.get("fiftyTwoWeekLow")
    mkt_cap = info.get("marketCap")
    pe_ratio = info.get("forwardPE")
    div_yield = info.get("dividendYield")
    today_stock=df.iloc[-1:]
    yesterday_stock=df.iloc[-2:]
    date_index=today_stock.index
    date_str = date_index[0].strftime('%Y-%m-%d')
    today_stock=today_stock.to_dict('records')[0]
    today_stock['Date']=date_str
    today_stock['fiftyTwoWeekHigh']=fifty_two_week_high
    today_stock['fiftyTwoWeekLow']=fifty_two_week_low
    today_stock['marketCap']=mkt_cap
    today_stock['forwardPE']=pe_ratio
    today_stock['dividendYield']=div_yield
    yesterday_stock=yesterday_stock.to_dict('records')[0]
    today_stock = {key: round(value, 2) if isinstance(value, float) else value for key, value in today_stock.items()}
    yesterday_stock = {key: round(value, 2) if isinstance(value, float) else value for key, value in yesterday_stock.items()}
    today_stock['openratio']=round(today_stock['Open']-yesterday_stock['Open'],2)
    today_stock['openratioper']=round(((today_stock['Open']-yesterday_stock['Open'])/yesterday_stock['Open'])*100,2)
    today_stock['highratio']=round(today_stock['High']-yesterday_stock['High'],2)
    today_stock['highratioper']=round(((today_stock['High']-yesterday_stock['High'])/yesterday_stock['High'])*100,2)
    today_stock['lowratio']=round(today_stock['Low']-yesterday_stock['Low'],2)
    today_stock['lowratioper']=round(((today_stock['Low']-yesterday_stock['Low'])/yesterday_stock['Low'])*100,2)
    today_stock['closeratio']=round(today_stock['Close']-yesterday_stock['Close'],2)
    today_stock['closeratioper']=round(((today_stock['Close']-yesterday_stock['Close'])/yesterday_stock['Close'])*100,2)
    today_stock['adjcloseratio']=round(today_stock['Adj Close']-yesterday_stock['Adj Close'],2)
    today_stock['adjcloseratioper']=round(((today_stock['Adj Close']-yesterday_stock['Adj Close'])/yesterday_stock['Adj Close'])*100,2)
    today_stock['volumeratio']=round(today_stock['Volume']-yesterday_stock['Volume'],2)
    today_stock['volumeratioper']=round(((today_stock['Volume']-yesterday_stock['Volume'])/yesterday_stock['Volume'])*100,2)
    company=COMPANY
    companymain=company.to_dict('records')
    company_df = company[company['Ticker'] == quote]
    company_df_dict=company_df.to_dict('records')[0]
    labels=['Overall','7day','15day','1month','3month','6month','1year']
    stock_symbol = quote
    market_symbol = '^GSPC'  # S&P 500 index symbol
    result={}
    beta = calculate_beta(stock_symbol, market_symbol, start, end)
    result['beta']=round(beta,2)
    if beta>1:
        result['interpretation']="Indicates that the stock is more volatile than the market"
    if beta<1:
        result['interpretation']="Indicates that the stock is stock is less volatile than the market"
    if beta==1:
        result['interpretation']="Indicates that the stock's volatility matches that of the market"
    return render_template("stat.html",today_stock=today_stock,company_dict=company_df_dict,labels=labels,COMPANY=companymain,beta=result)


@app.route('/fetchstock', methods=['GET'])
def data():
    ticker = request.args.get('ticker')  # Get the ticker from the request URL query parameters
    df = pd.read_csv('data/individual/'+ticker+'.csv')
    df['timestamp']= df['Date'].astype(str)
    # Convert datetime objects to timestamps
    # df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
    newdf=df[['timestamp','Close']]
    result = newdf.values.tolist()
    company=COMPANY
    data_dict=company['Ticker'].tolist()
    if ticker in data_dict:
        return jsonify({'ticker': ticker, 'data': result})
    else:
        return jsonify({'error': 'Ticker not found'})

@app.route('/arimamodel', methods=['GET'])
def arimamodels():
    ticker = request.args.get('ticker')  # Get the ticker from the request URL query parameters
    df = pd.read_csv('data/individual/'+ticker+'.csv')
    df = df.dropna()
    code_list=[]
    for i in range(0,len(df)):
        code_list.append(ticker)
    df2=pd.DataFrame(code_list,columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df=df2
    arima_main=ARIMA_ALGO(df,forecast_days)
    company=COMPANY
    data_dict=company['Ticker'].tolist()
    if ticker in data_dict:
        return jsonify({'ticker': ticker, 'data': arima_main})
    else:
        return jsonify({'error': 'Ticker not found'})

@app.route('/regmodel', methods=['GET'])
def regmodels():
    ticker = request.args.get('ticker')  # Get the ticker from the request URL query parameters
    print(ticker)
    df = pd.read_csv('data/individual/'+ticker+'.csv')
    df = df.dropna()
    code_list=[]
    for i in range(0,len(df)):
        code_list.append(ticker)
    df2=pd.DataFrame(code_list,columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df=df2
    reg_main=LIN_REG_ALGO(df,forecast_days)
    company=COMPANY
    data_dict=company['Ticker'].tolist()
    if ticker in data_dict:
        return jsonify({'ticker': ticker, 'data': reg_main})
    else:
        return jsonify({'error': 'Ticker not found'})

@app.route('/svmmodel', methods=['GET'])
def svmmodels():
    ticker = request.args.get('ticker')  # Get the ticker from the request URL query parameters
    print(ticker)
    df = pd.read_csv('data/individual/'+ticker+'.csv')
    df = df.dropna()
    code_list=[]
    for i in range(0,len(df)):
        code_list.append(ticker)
    df2=pd.DataFrame(code_list,columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df=df2
    reg_main=SVM_ALGO(df,forecast_days)
    company=COMPANY
    data_dict=company['Ticker'].tolist()
    if ticker in data_dict:
        return jsonify({'ticker': ticker, 'data': reg_main})
    else:
        return jsonify({'error': 'Ticker not found'})


if __name__ == "__main__":
	app.run()