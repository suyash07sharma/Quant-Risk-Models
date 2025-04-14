# models/arma_garch.py

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t, gamma
from scipy.special import gamma as gamma_function

class ARMAGARCH:
    """
    Implementation of ARMA(1,1)-GARCH(1,1) model with Student-T innovations.
    """
    
    def __init__(self, p=1, q=1, alpha=0.05, window_size=250):
        """Initialize the ARMA-GARCH model."""
        self.p = p
        self.q = q
        self.alpha = alpha
        self.window_size = window_size
        self.params = None
        self.fitted_variances = None
        self.fitted_residuals = None
        self.fitted_innovations = None
        
    def log_likelihood(self, params, returns):
        """Calculate the log-likelihood of ARMA-GARCH model."""
        phi, theta, psi, beta, nu = params
        
        # Parameter constraints
        if psi + beta >= 1 or psi <= 0 or beta <= 0 or nu <= 2:
            return 1e10
        
        n = len(returns)
        eps = np.zeros(n)
        sigma2 = np.zeros(n)
        
        # Calculate unconditional variance for initialization
        unconditional_variance = np.var(returns)
        omega = unconditional_variance * (1 - psi - beta)
        
        if omega <= 0:
            return 1e10
            
        sigma2[0] = unconditional_variance
        
        # ARMA-GARCH process
        for t in range(1, n):
            mu_t = phi * returns[t-1]
            
            if t > 1:
                mu_t += theta * eps[t-1]
                
            eps[t] = returns[t] - mu_t
            sigma2[t] = omega + psi * sigma2[t-1] + beta * eps[t-1]**2
            
        # Student-T log-likelihood
        v = 0.5 * (nu + 1)
        log_likelihood = n * (np.log(gamma_function(v)) - np.log(gamma_function(0.5 * nu)) 
                             - 0.5 * np.log(np.pi * (nu - 2)))
        
        for t in range(1, n):
            log_likelihood -= 0.5 * np.log(sigma2[t])
            log_likelihood -= v * np.log(1 + (eps[t]**2) / (sigma2[t] * (nu - 2)))
            
        return -log_likelihood
    
    def fit(self, returns):
        """Fit the ARMA-GARCH model using rolling window."""
        n = len(returns)
        result_params = []
        fitted_variances = []
        fitted_residuals = []
        fitted_innovations = []
        
        for i in range(self.window_size, n):
            window = returns[i-self.window_size:i]
            
            # Initial parameter guess
            initial_params = [0.1, 0.1, 0.3, 0.6, 8.0]
            
            # Parameter bounds
            bounds = [
                (-0.99, 0.99),  # phi
                (-0.99, 0.99),  # theta
                (0.01, 0.3),    # psi
                (0.01, 0.97),   # beta
                (2.1, 30.0)     # nu
            ]
            
            # Optimization
            result = minimize(
                self.log_likelihood,
                initial_params,
                args=(window,),
                bounds=bounds,
                method='COBYLA'
            )
            
            result_params.append(result.x)
            
            # Calculate forecasts
            phi, theta, psi, beta, nu = result.x
            unconditional_variance = np.var(window)
            omega = unconditional_variance * (1 - psi - beta)
            
            eps_last = 0
            sigma2_last = unconditional_variance
            
            if len(fitted_residuals) > 0:
                eps_last = fitted_residuals[-1]
                sigma2_last = fitted_variances[-1]
                
            mu_next = phi * window[-1] + theta * eps_last
            sigma2_next = omega + psi * sigma2_last + beta * eps_last**2
            
            fitted_variances.append(sigma2_next)
            
            if i < n:
                eps_next = returns[i] - mu_next
                fitted_residuals.append(eps_next)
                fitted_innovations.append(eps_next / np.sqrt(sigma2_next))
            
        self.params = np.array(result_params)
        self.fitted_variances = np.array(fitted_variances)
        self.fitted_residuals = np.array(fitted_residuals)
        self.fitted_innovations = np.array(fitted_innovations)
        
        return self
    
    def forecast_var(self, alpha=None):
        """Forecast Value-at-Risk."""
        if alpha is None:
            alpha = self.alpha
            
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")
        
        var_forecasts = []
        
        for i, param_set in enumerate(self.params):
            _, _, _, _, nu = param_set
            sigma = np.sqrt(self.fitted_variances[i])
            
            # Calculate Student-T quantile
            t_quantile = t.ppf(alpha, nu)
            
            # Calculate VaR
            var_forecast = t_quantile * sigma
            var_forecasts.append(var_forecast)
            
        return np.array(var_forecasts)
