import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import timedelta
from datetime import datetime
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from scipy import stats
import seaborn as sns
import yfinance as yf
import math

#-----------TODO -------------
# 1. Plot the Efficient Frontier witha CLA object
# 2. Implement the min_volatilty optimization in optimizePortSharpe()
# 3. Implement the new function for using the CLA optimizer solution

#-----------------------------

#Taking an example portfolio of FAANG
def getData(portfolio, start, end = datetime.today().strftime('%Y-%m-%d')):
    df = pd.DataFrame()
    #Getting the data
    for stock in portfolio:
        df[stock] = web.DataReader(stock, data_source='yahoo', start = start, end = end)['Adj Close']
    return df
def plotPort(df, port):
    for stock in portfolio:
        plt.plot(df[stock], label=stock)
        plt.title("Stocks ove given period")
        plt.legend()
        plt.show()

def basicStats(df, weights, start):
    #Calculating the essential Values for the uder entered portfolio
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_volatility = np.sqrt(port_variance)
    annual_return = np.sum(returns.mean()*weights) * 252

    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_vols = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(annual_return, 2)*100)+'%'

    df = df.pct_change()[1:]
    df_spy = web.DataReader('SPY', data_source='yahoo', start = start, end = datetime.today().strftime('%Y-%m-%d'))['Adj Close']
    df_spy = df_spy.pct_change()[1:]
    port_ret = (df * weights).sum(axis = 1)
    (beta, alpha) = stats.linregress(df_spy.values, port_ret.values)[0:2]
                
    #This prints the stats for the portfolio passed in by the user
    print("The basic stats of the portfolio: ")
    print("Expected annual return : ", percent_ret)
    print('Annual volatility/standard deviation/risk : ',percent_vols)
    print('Annual variance : ',percent_var)
    print("Portfolio Beta :", round(beta, 4))

def getDiscreteAllocations(df, weights):
    latest_prices = get_latest_prices(df)
    #plotting.plot_weights(weights)
    total_portfolio_value  = 15000
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()

    print("Best portfolio possible today for the given shares and given contraints for an investment of ", total_portfolio_value, ": ")
    print("Shares allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))

def VaR(portfolio, weights, price, date = datetime.today()):
    df = pd.DataFrame()
    for stock in portfolio:
        s = yf.Ticker(stock)
        df[stock] = s.history(period='max')["Close"]
    df = df[-501:]
    df_exp =(df)/df.iloc[0]
    df_exp = df_exp*weights*price
    df['Value'] = df_exp.sum(axis = 1)
    df_loss = df.set_index(np.arange(0,501,1))
    for i in range(1,501):
        df_loss.iloc[i-1] = (df.iloc[i]/df.iloc[i-1])*df.iloc[-1]
    df_loss = df_loss[:-1]
    for i in range (500):
        df_loss['Value'].iloc[i]  = round(df_loss["Value"].iloc[i]-df["Value"].iloc[-1] , 2)
    arr = df_loss['Value'].values *-1
    arr = np.sort(arr)
    print("The 1 day 99 percent confidence VaR is: ",'{:2f}'.format(round(arr[4],2)*-1))
    print("The 10 day 99 percent confidence VaR is: ",'{:2f}'.format(round(arr[4],2)*math.sqrt(10)*-1))


   

# portfolio = ['FB', "AAPL", "AMZN", 'NFLX', 'GOOG']
# weights = np.array([0.2,0.2,0.2,0.2,0.2])
# start = '2013-01-01'
# price = 10000000
# VaR(portfolio, weights, price)
