import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import CLA
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


#-----------TODO -------------
# 1. Plot the Efficient Frontier witha CLA object
# 2. Implement the min_volatilty optimization in optimizePortSharpe()
# 3. Implement the new function for using the CLA optimizer solution

#-----------------------------

#Taking an example portfolio of FAANG
def getData(portfolio, start):
    df = pd.DataFrame()
    today = datetime.today().strftime('%Y-%m-%d')
    #Getting the data
    for stock in portfolio:
        df[stock] = web.DataReader(stock, data_source='yahoo', start = start, end = today)['Adj Close']
    return df
def plotPort(df, port):
    for stock in portfolio:
        plt.plot(df[stock], label=stock)
        plt.title("Stocks ove given period")
        plt.legend()
        plt.show()

def basicStats(df, weights):
    #Calculating the essential Values for the uder entered portfolio
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_volatility = np.sqrt(port_variance)
    annual_return = np.sum(returns.mean()*weights) * 252

    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_vols = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(annual_return, 2)*100)+'%'

    #This prints the stats for the portfolio passed in by the user
    print("The basic stats of the portfolio: ")
    print("Expected annual return : ", percent_ret)
    print('Annual volatility/standard deviation/risk : ',percent_vols)
    print('Annual variance : ',percent_var)

def getDiscreteAllocations(df, weights):
    latest_prices = get_latest_prices(df)
    #plotting.plot_weights(weights)
    total_portfolio_value  = 15000
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()

    print("Best portfolio possible today for the given shares and given contraints for an investment of ", total_portfolio_value, ": ")
    print("Shares allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))

#Plotting the efficient frontier
