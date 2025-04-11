# Financial Risk Model Evaluation

## Overview
This project evaluates the performance of ARMA-GARCH models with Student-T innovations for risk prediction across diverse asset classes. The implementation uses Python for statistical analysis and C++ for high-performance components, focusing on Value-at-Risk (VaR) backtesting and distributional accuracy.

![Risk Model Performance](results/spy_var_performance.png)

## Features
- Implementation of ARMA(1,1)-GARCH(1,1) models with Student-T innovations
- Rolling window parameter estimation with maximum likelihood
- VaR and CVaR calculation and backtesting at 95% and 99% confidence levels
- Statistical validation using Kupiec test, Christoffersen test, and Kolmogorov-Smirnov test
- Cross-asset class performance comparison

## Asset Classes Analyzed
- Equities (SPY - S&P 500 ETF)
- Real Estate (IYR - iShares U.S. Real Estate ETF)
- Gold (GLD - SPDR Gold Shares)
- Oil (USO - United States Oil Fund)
- Treasury Bonds (TLT - iShares 20+ Year Treasury Bond ETF)

## Key Findings
- The model demonstrates strong performance for equities, real estate, and gold with acceptable p-values for VaR predictions
- Heavy-tailed Student-T distribution effectively captures extreme market movements
- Volatility persistence is consistently high across all asset classes
- Commodities (particularly oil) require specialized modeling approaches
- The model passes regulatory standards for most asset classes

## Technical Implementation
- **Data Processing**: Pandas for data preparation, NumPy for numerical operations
- **Parameter Estimation**: Maximum likelihood estimation with COBYLA optimizer
- **Statistical Testing**: Implementation of Kupiec, Christoffersen, and Kolmogorov-Smirnov tests
- **Visualization**: Custom matplotlib plots for time series, QQ plots, and VaR breaches

## Usage
```python
# Example of model fitting and evaluation
from models.garch import ARMAGARCH
from analysis.backtesting import var_exceedance_test

# Initialize model with parameters
model = ARMAGARCH(p=1, q=1, window_size=250)

# Fit model and generate forecasts
model.fit(returns_data)
var_predictions = model.forecast_var(alpha=0.05)

# Evaluate performance
results = var_exceedance_test(actual_returns, var_predictions, alpha=0.05)
