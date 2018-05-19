import os
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from backend.config import currencies

file = 'history/AUD_JPY_D.csv'
folder = 'history/'

def monteCarloSimulation(file, n_days, n_iters):
    """
    file: csv file containing at least a 'Close' column with asset prices inside
    n_days: the number of days to simulate retruns for
    n_iters: the number of different simulations to run

    returns the price_matrix of the simulation
    """
    # import data
    df = pd.read_csv(file)
    prices = df['Close']
    # set formula's variables
    log_returns = np.log(1 + prices.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    std = log_returns.std()
    drift = pd.Series(drift)
    std = pd.Series(std)
    z = norm.ppf(np.random.rand(n_days, n_iters))
    # generate future daily returns
    daily_returns = np.exp(drift.values + std.values * z)
    p0 = prices.iloc[-1] # last price in our original data is the first price in our simulation
    price_matrix = np.zeros_like(daily_returns)
    price_matrix[0] = p0
    # start the simulation
    for t in range(1, n_days):
        price_matrix[t] = price_matrix[t-1] * daily_returns[t]
    plt.plot(price_matrix)
    plt.show()

    return price_matrix

# Brownian motion
def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1/N  # timestep
    b = np.random.normal(0., 1., int(N)) * np.sqrt(dt) # Brownian normal increments
    W = np.cumsum(b) #Brownian path
    return W, b

# Geometric Brownian Motion
def GBM(p0, mu, sigma, W, T, N):
    t = np.linspace(0., 1., N+1)
    P = list()
    P.append(p0)
    for i in range(1, int(N + 1)):
        drift = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W[i-1]
        p_temp = p0 * np.exp(drift + diffusion)
        P.append(p_temp)
    return P, t

# Euler Maruyama Approximation
def eulerApprox(p0, mu, sigma, b, T, N, M):
    dt = M * (1/N) # step size
    L = N / M
    wi = [p0]
    for i in range(0, int(L)):
        Winc = np.sum(b[(M * (i - 1) + M) : (M * i + M)])
        w_i_new = wi[i] + mu * wi[i] * dt + sigma * wi[i] * Winc
        wi.append(w_i_new)
    return wi, dt

# plot these stochastic compared to a given asset
def plot_stochastics(file, seed1, seed2, seed3):
    df = pd.read_csv(file)
    n = len(df) - 1
    returns = df['Close'].pct_change()
    p0 = df['Close'][n]
    N = 2.0 ** 6
    W1 = Brownian(seed1, N)[0]
    W2 = Brownian(seed2, N)[0]
    W3 = Brownian(seed3, N)[0]
    T = 1.0
    mu = np.mean(returns) * 252.0
    sigma = np.std(returns) * np.sqrt(252.0)

    gbm1 = GBM(p0, mu, sigma, W1, T, N)[0]
    gbm2 = GBM(p0, mu, sigma, W2, T, N)[0]
    gbm3 = GBM(p0, mu, sigma, W3, T, N)[0]
    t = GBM(p0, mu, sigma, W1, T, N)[1]

    plt.plot(t, gbm1, label='GMB1', ls='--')
    plt.plot(t, gbm2, label='GBM2', ls='--')
    plt.plot(t, gbm3, label='GBM3', ls='--')
    plt.plot(t, df['Close'][-65:], label='Actual')
    plt.ylabel('Asset Price, $')
    plt.title('Geometric Brownian Motion')
    plt.legend(loc='upper left')
    plt.show()

# Markowitz Portfolio optimization, efficient frontier, MPT, etc.
def optimalPortfolio(asset_list, n_portfolios=2500):
    """
    asset_list: list of asset symbols to extract
    type: CSV or GET; CSV from local file or GET from yahoo
    """
    data = pd.DataFrame()
    # get asset pricing from local data
    if type == 'csv':
        for asset in asset_list:
            folder = 'history/'
            name_window = '%s_D' % asset
            for files in os.walk(folder):
                for f in files[2]:
                    if name_window in f:
                        f = os.path.join(folder, f)
                        df = pd.read_csv(f)
                        data[asset] = df['Close']
    elif type == 'GET':
        for asset in asset_list:
            s = stock(asset).chart_table(range='1y')
            data[asset] = s['close']

    # convert prices to returns and calculate daily mean returns and covariance
    returns = data.pct_change()
    n = returns.shape[1]
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    # set array to hold results
    results = np.zeros((3+n, n_portfolios))
    # calculate the n portfolios
    for i in range(n_portfolios):
        # set random weights; rebalance weights to sum 1
        weights = np.random.random(n)
        weights /= np.sum(weights)
        # calculate portfolio return and volatility
        port_return = np.sum(mean_daily_returns * weights) * 252
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # store results and calculate/store Sharpe ratio
        results[0,i] = port_return
        results[1,i] = port_std
        results[2,i] = results[0,i] / results[1,i]
        # iterate through weight vector and add weights to the results
        for j in range(len(weights)):
            results[j+3,i] = weights[j]
    # plot results
    cols = ['Returns', 'Std', 'Sharpe'] + asset_list
    results_frame = pd.DataFrame(results.T, columns=cols)
    max_sharpe_port = results_frame.iloc[results_frame['Sharpe'].idxmax()]
    min_vol_port = results_frame.iloc[results_frame['Std'].idxmin()]
    plt.scatter(results_frame.Std, results_frame.Returns, c=results_frame.Sharpe, cmap='RdYlBu')
    plt.colorbar()
    plt.ylabel('Returns')
    plt.xlabel('Volatility')
    plt.title('Portfolio optimization test for %d portfolios with %d assets' % (n_portfolios, n))
    plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5,1,0), color='r', s=1000)
    plt.scatter(min_vol_port[1], min_vol_port[0], marker=(5,1,0), color='g', s=1000)
    plt.show()
    print("Portfolio weights with the best returns: ", max_sharpe_port)
    print("Portfolio weights with the lowest volatility: ", min_vol_port)
