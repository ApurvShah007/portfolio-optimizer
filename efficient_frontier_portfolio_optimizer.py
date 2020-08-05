import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting

#Importing my own file to use the functions:
import basic_portfolio_functions as bpf

#The Efficient Frontier Method to optimize
def optimizePortEfficient(port, weights, start, plot = False, short = False, printBasicStats=True, how = 'Sharpe'):
    #Getting Data
    df = bpf.getData(port, start)
    #Plotting the portfolio
    if plot: 
        bpf.plotPort(df, port)
        
    if printBasicStats:
        bpf.basicStats(df, weights, start)
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
        bpf.getDiscreteAllocations(df, weights)
        plotting.plot_weights(weights)
        return weights 
    elif how == "Vol":
        # Minimized on Volatility
        efi = EfficientFrontier(mu, S, weight_bounds=bounds)
        w = dict(efi.min_volatility())
        print("Weights of an optimal portfolio minimized on Volatilty (Risk):")
        print(w)
        efi.portfolio_performance(verbose = True)
        bpf.getDiscreteAllocations(df, w)
        plotting.plot_weights(w)
        return w
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
            bpf.getDiscreteAllocations(df, w)
            plotting.plot_weights(w)
        return w

# an Example FAANG portfolio with equal weights
# portfolio = ['FB', "AAPL", "AMZN", 'NFLX', 'GOOG']
# weights = np.array([0.2,0.2,0.2,0.2,0.2])
# start = '2013-01-01'
# price = 10000000
# w = optimizePortEfficient(portfolio, weights, start)
# w = list(w.values())

# bpf.VaR(portfolio, weights, price)
# bpf.VaR(portfolio, w, price)

