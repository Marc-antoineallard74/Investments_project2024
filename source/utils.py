import numpy as np
import matplotlib.pyplot as plt

def annualized_metrics(returns):
    mean_return = returns.mean() * 12
    std_dev = returns.std() * np.sqrt(12)
    sharpe_ratio = mean_return / std_dev
    return mean_return, std_dev, sharpe_ratio

def  plot_strategy_performance(data, metrics, columns):
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))


    for i, metric in enumerate(metrics):
        ax = axes[i]
        data.plot(kind='bar', x='Metric', y=[columns],
                    ax=ax, legend=True)
        ax.set_title(metric)
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.show()
