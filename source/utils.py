import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

def annualized_metrics(returns):
    mean_return = returns.mean() * 12
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
def ew_strategy(data, columns):
    strategy_returns = data[columns].mean(axis=1)
    return strategy_returns

# Value weight strategy
def vw_strategy(data, columns):
    strategy_returns = data[columns].apply(lambda x: np.average(x['Rn'], weights=x['mcap']), axis=1)
    return strategy_returns

# Risk parity strategy
def rp_strategy(data, columns):
    volatilities = data[columns].rolling(window=12).std()
    inverse_vols = 1 / volatilities
    weights = inverse_vols.div(inverse_vols.sum(axis=1), axis=0)
    strategy_returns = (weights * data[columns]).sum(axis=1)
    return strategy_returns

# Mean-variance optimization strategy
def mv_strategy(data, columns):
    returns = data[columns]
    mean_returns = returns.rolling(window=12).mean()
    cov_matrix = returns.rolling(window=12).cov()
    
    def get_optimal_weights(mean_returns, cov_matrix, lambda_=0.1):
        cov_matrix += lambda_ * np.eye(len(cov_matrix))  # Regularization
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(mean_returns))
        weights = inv_cov.dot(ones) / ones.dot(inv_cov).dot(ones)
        return weights

    optimal_weights = mean_returns.apply(lambda x: get_optimal_weights(x, cov_matrix.loc[x.name]), axis=1)
    strategy_returns = (optimal_weights * returns).sum(axis=1)
    return strategy_returns

# ---------------------------------- Scaling --------------------------------- #
def scale_to_target_volatility(strategy_returns, target_vol=0.10):
    annual_vol = strategy_returns.std() * np.sqrt(12)
    scale_factor = target_vol / annual_vol
    return strategy_returns * scale_factor