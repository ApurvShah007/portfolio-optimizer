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
def getData(portfolio):
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

def plot_ef(cla, points=100, show_assets=True):
    if cla.weights is None:
        cla.max_sharpe()
    optimal_ret, optimal_risk, _ = cla.portfolio_performance()

    if cla.frontier_values is None:
        cla.efficient_frontier(points=points)

    mus, sigmas, _ = cla.frontier_values

    fig, ax = plt.subplots()
    ax.plot(sigmas, mus, label="Efficient frontier")

    if show_assets:
        ax.scatter(
            np.sqrt(np.diag(cla.cov_matrix)),
            cla.expected_returns,
            s=30,
            color="k",
            label="assets",
        )

    ax.scatter(optimal_risk, optimal_ret, marker="x", s=100, color="r", label="optimal")
    ax.legend()
    ax.set_title("Efficient Frontier Plot")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    plt.show()

#The Critical Line Algorithm method of portfolio optimization
def omptimizePortCLA(port, weights, start, plot = False, short = False, printBasicStats=True, how = 'Sharpe'):
    #Getting Data
    df = getData(port)
    #Plotting the portfolio
    if plot: 
        plotPort(df, port)
        
    if printBasicStats:
        basicStats(df, weights)
    #Optimization for Sharpe using Efficient Frontier
    if short: 
        bounds = (-1,1)
    else:
        bounds = (0,1)
    mu = df.pct_change().mean() * 252
    S = risk_models.sample_cov(df)
    cla = CLA(mu, S)
    cla.max_sharpe()
    w = dict(cla.clean_weights())
    print(w)
    cla.portfolio_performance(verbose = True)
    getDiscreteAllocations(df, w)
    plot_ef(cla)
    plotting.plot_weights(w)


#The Efficient Frontier Method to optimize
def optimizePortEfficient(port, weights, start, plot = False, short = False, printBasicStats=True, how = 'Sharpe'):
    #Getting Data
    df = getData(port)
    #Plotting the portfolio
    if plot: 
        plotPort(df, port)
        
    if printBasicStats:
        basicStats(df, weights)
    #Optimization for Sharpe using Efficient Frontier
    if short: 
        bounds = (-1,1)
    else:
        bounds = (0,1)
    mu = df.pct_change().mean() * 252
    S = risk_models.sample_cov(df)
    #Method and constraints for optimization
    if how == 'Sharpe':
        # Maximized on Sharpe Ratio
        ef = EfficientFrontier(mu, S, weight_bounds=bounds) #Here the weight bounds are being used to allow short positions as well
        weights = ef.max_sharpe()
        cleaned_weights = dict(ef.clean_weights())
        print("Weights of an optimal portfolio maximised on Sharpe Ratio:")
        print(cleaned_weights)
        ef.portfolio_performance(verbose = True)
        getDiscreteAllocations(df, weights)
        plotting.plot_weights(weights)
    elif how == "Vol":
        # Minimized on Volatility
        efi = EfficientFrontier(mu, S, weight_bounds=bounds)
        w = dict(efi.min_volatility())
        print("Weights of an optimal portfolio minimized on Volatilty (Risk):")
        print(w)
        efi.portfolio_performance(verbose = True)
        getDiscreteAllocations(df, w)
        plotting.plot_weights(w)
    elif how == "targetRisk":
        #Optimized for a given target risk
        efi = EfficientFrontier(mu, S, weight_bounds=bounds)
        efi.efficient_risk(0.25)
        w = dict(efi.clean_weights())
        if w ==None:
            print("No portfolio possible at the given risk level")
        else:
            print("Weights of an optimal portfolio for given risk:")
            print(w)
            efi.portfolio_performance(verbose = True)
            getDiscreteAllocations(df, w)
            plotting.plot_weights(w)

# an Example FAANG portfolio with equal weights
portfolio = ['FB', "AAPL", "AMZN", 'NFLX', 'GOOG']
weights = np.array([0.2,0.2,0.2,0.2,0.2])
start = '2013-01-01'
omptimizePortCLA(portfolio, weights, start, how = "Vol")

