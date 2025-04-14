import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def plot_returns_and_var(returns, var_values, ticker, alpha=0.05):
    """
    Plot returns with VaR lines and highlight exceedances.
    
    Parameters:
        returns (array_like): Actual returns
        var_values (array_like): Predicted VaR values
        ticker (str): Asset ticker for the plot title
        alpha (float): VaR significance level
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot returns
    ax.scatter(range(len(returns)), returns, s=10, color='blue', alpha=0.7, label='Actual Returns')
    
    # Plot VaR
    ax.plot(range(len(var_values)), var_values, color='red', linewidth=1.5, label=f'{int((1-alpha)*100)}% VaR')
    
    # Highlight exceedances
    exceedances = returns < var_values
    ax.scatter(np.where(exceedances)[0], returns[exceedances],
               color='red', s=50, marker='x', label='VaR Exceedances')
    
    # Add labels and title
    ax.set_xlabel('Observation')
    ax.set_ylabel('Return')
    ax.set_title(f'{ticker} - Returns vs. VaR({int((1-alpha)*100)}%)')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_qq(model_quantiles, ticker):
    """
    Create a QQ plot comparing model quantiles to uniform quantiles.
    
    Parameters:
        model_quantiles (array_like): Model-transformed quantiles
        ticker (str): Asset ticker for the plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Sort the model quantiles
    sorted_quantiles = np.sort(model_quantiles)
    
    # Create uniform quantiles
    n = len(sorted_quantiles)
    uniform_quantiles = np.arange(1, n + 1) / (n + 1)
    
    # Calculate confidence bands (Kolmogorov-Smirnov 95% confidence bands)
    conf_band = 1.36 / np.sqrt(n)
    upper_band = np.minimum(uniform_quantiles + conf_band, 1)
    lower_band = np.maximum(uniform_quantiles - conf_band, 0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the points
    ax.scatter(uniform_quantiles, sorted_quantiles, s=15, color='blue', alpha=0.7)
    
    # Plot the 45-degree line
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5)
    
    # Plot confidence bands
    ax.plot(uniform_quantiles, upper_band, 'g--', linewidth=1, alpha=0.5)
    ax.plot(uniform_quantiles, lower_band, 'g--', linewidth=1, alpha=0.5)
    
    # Fill between the bands
    ax.fill_between(uniform_quantiles, lower_band, upper_band, color='green', alpha=0.1)
    
    # Add labels and title
    ax.set_xlabel('Uniform Quantiles')
    ax.set_ylabel('Model Quantiles')
    ax.set_title(f'QQ Plot for {ticker}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_parameter_evolution(params, param_names, ticker):
    """
    Plot the evolution of model parameters over time.
    
    Parameters:
        params (array_like): Array of parameter sets
        param_names (list): Names of the parameters
        ticker (str): Asset ticker for the plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    n_params = len(param_names)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params), sharex=True)
    
    # Plot each parameter
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(params[:, i], linewidth=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Parameter Evolution for {ticker}', fontsize=16)
    
    # Add x-label to the bottom subplot
    axes[-1].set_xlabel('Observation')
    
    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

def plot_comparison_bar(results_dict, metric='p_value', title='Model Comparison'):
    """
    Create a bar chart comparing results across assets.
    
    Parameters:
        results_dict (dict): Dictionary mapping assets to result dictionaries
        metric (str): The metric to compare
        title (str): The plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    assets = list(results_dict.keys())
    values = [results[metric] for results in results_dict.values()]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the bars
    bars = ax.bar(assets, values, width=0.6, alpha=0.7)
    
    # Add a horizontal line at 0.05 if the metric is p-value
    if metric == 'p_value':
        ax.axhline(y=0.05, color='r', linestyle='--', linewidth=1.5)
        
        # Color the bars based on acceptance/rejection
        for i, bar in enumerate(bars):
            if values[i] > 0.05:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Add a legend
        legend_elements = [
            Line2D([0], [0], color='green', lw=4, label='Accepted'),
            Line2D([0], [0], color='red', lw=4, label='Rejected'),
            Line2D([0], [0], color='r', linestyle='--', lw=1.5, label='5% Significance')
        ]
        ax.legend(handles=legend_elements, loc='best')
    
    # Add labels and title
    ax.set_xlabel('Asset')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of the bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    return fig
