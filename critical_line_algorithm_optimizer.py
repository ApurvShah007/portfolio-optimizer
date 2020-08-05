import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import CLA
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting

#Importing my own file to use the functions:
import basic_portfolio_functions as bpf


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
            label="assets",)
    ax.scatter(optimal_risk, optimal_ret, marker="x", s=100, color="r", label="optimal")
    ax.legend()
    ax.set_title("Efficient Frontier Plot")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    plt.show()


#The Critical Line Algorithm method of portfolio optimization
def omptimizePortCLA(port, weights, start, plot = False, short = False, printBasicStats=True, how = 'Sharpe'):
    #Getting Data
    df = bpf.getData(port, start)
    #Plotting the portfolio
    if plot: 
        bpf.plotPort(df, port)
        
    if printBasicStats:
        bpf.basicStats(df, weights)
    #Optimization for Sharpe using Efficient Frontier
    if short: 
        bounds = (-1,1)
    else:
        bounds = (0,1) 
    mu = df.pct_change().mean() * 252
    S = risk_models.sample_cov(df)

    if how == 'Sharpe':
        # Maximized on Sharpe Ratio
        cla = CLA(mu, S) #Here the weight bounds are being used to allow short positions as well
        weights = cla.max_sharpe()
        cleaned_weights = dict(cla.clean_weights())
        print("Weights of an optimal portfolio maximised on Sharpe Ratio:")
        print(cleaned_weights)
        cla.portfolio_performance(verbose = True)
        bpf.getDiscreteAllocations(df, weights)
        plot_ef(cla)
        plotting.plot_weights(weights)
    elif how == "Vol":
        # Minimized on Volatility
        cla = CLA(mu, S)
        cla.min_volatility()
        w = dict(cla.clean_weights())
        print("Weights of an optimal portfolio minimized on Volatilty (Risk):")
        print(w)
        cla.portfolio_performance(verbose = True)
        bpf.getDiscreteAllocations(df, w)
        plot_ef(cla)
        plotting.plot_weights(w)

#an Example FAANG portfolio with equal weights
# portfolio = ['FB', "AAPL", "AMZN", 'NFLX', 'GOOG']
# weights = np.array([0.2,0.2,0.2,0.2,0.2])
# start = '2013-01-01'



