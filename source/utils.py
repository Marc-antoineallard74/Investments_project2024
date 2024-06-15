import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

def annualized_metrics(returns, rf):
    mean_return = (returns-rf).mean() * 12
    std_dev = returns.std() * np.sqrt(12)
    sharpe_ratio = mean_return / std_dev
    return mean_return, std_dev, sharpe_ratio

# ----------------------------------- Plots ---------------------------------- # 

def plot_strategy_performance(data, metrics, columns):
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))


    for i, metric in enumerate(metrics):
        ax = axes[i]
        data.plot(kind='bar', x='Metric', y=columns,
                    ax=ax, legend=True)
        ax.set_title(metric)
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.show()

def plot_metrics(returns, title):
    # Calculate annualized mean, standard deviation, and Sharpe ratio
    metrics = returns.apply(annualized_metrics, axis=0)

    # As DataFrame
    metrics_df = metrics.T.rename(columns={0: 'Mean', 1: 'Volatility', 2: 'Sharpe Ratio'})

    if len(metrics_df) > len(metrics):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))

    for i, m in enumerate(metrics_df.columns):
        sns.barplot(metrics_df[m], ax=axes[i])
        axes[i].set_title(m)
        #axes[i].set_xlabel(m)
        axes[i].set_ylabel(m)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# --------------------------- Weighting Strategies --------------------------- #

# Equal weight strategy
def ew_strategy(data, asset_columns):
    strategy_returns = data[asset_columns].mean(axis=1)
    return strategy_returns

# Value weight strategy
def vw_strategy(data, asset_columns):
    strategy_returns = data[asset_columns].apply(lambda x: np.average(x['Rn'], weights=x['mcap']), axis=1)
    return strategy_returns

# Risk parity strategy
def rp_strategy(data, asset_columns):
    volatilities = data[asset_columns].rolling(window=12).std()
    inverse_vols = 1 / volatilities
    weights = inverse_vols.div(inverse_vols.sum(axis=1), axis=0)
    strategy_returns = (weights * data[asset_columns]).sum(axis=1)
    return strategy_returns

# Mean-variance optimization strategy
def mv_strategy(data, columns):
    returns = data[columns]

    mu = returns.mean()
    rf = data['rf'].mean()

    sigma = returns.cov()
    sigma_inv = np.linalg.inv(sigma)

    A = np.sum(sigma_inv)
    B = np.sum(sigma_inv @ mu)

    w_tan = sigma_inv @ (mu - rf) / (B - A*rf)

    strategy_returns = returns @ w_tan
    return strategy_returns

# ---------------------------------- Scaling --------------------------------- #
def scale_to_target_volatility(strategy_returns, target_vol=0.10):
    annual_vol = strategy_returns.std() * np.sqrt(12)
    scale_factor = target_vol / annual_vol
    return strategy_returns * scale_factor