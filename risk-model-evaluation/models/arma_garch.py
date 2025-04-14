## 2. models/arma_garch.py

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t

class ARMAGARCH:
    """
    Implementation of ARMA(1,1)-GARCH(1,1) model with Student-T innovations.
    
    This class implements a first-order Autoregressive Moving Average model
    for returns, coupled with a Generalized Autoregressive Conditional
    Heteroskedasticity model for volatility, with Student-T distributed innovations.
    
    Attributes:
        p (int): AR order
        q (int): MA order
        alpha (float): Significance level for VaR calculation
        window_size (int): Rolling window size for parameter estimation
        params (numpy.ndarray): Fitted model parameters
    """
    
    def __init__(self, p=1, q=1, alpha=0.05, window_size=250):
        """
        Initialize the ARMA-GARCH model.
        
        Parameters:
            p (int): AR order
            q (int): MA order
            alpha (float): Significance level for VaR calculation
            window_size (int): Rolling window size for parameter estimation
        """
        self.p = p
        self.q = q
        self.alpha = alpha
        self.window_size = window_size
        self.params = None
        self.fitted_returns = None
        self.fitted_variances = None
        self.fitted_residuals = None
        self.fitted_innovations = None
        
    def log_likelihood(self, params, returns):
        """
        Calculate the log-likelihood of the ARMA-GARCH model.
        
        Parameters:
            params (array_like): Model parameters [phi, theta, psi, beta, nu]
            returns (array_like): Return series
            
        Returns:
            float: Negative log-likelihood (for minimization)
        """
        phi, theta, psi, beta, nu = params
        
        # Parameter constraints to ensure stationarity
        if psi + beta >= 1 or psi <= 0 or beta <= 0 or nu <= 2:
            return 1e10  # Return a large value for invalid parameters
        
        # Initialize arrays
        n = len(returns)
        eps = np.zeros(n)
        sigma2 = np.zeros(n)
        
        # Calculate unconditional variance for initialization
        unconditional_variance = np.var(returns)
        omega = unconditional_variance * (1 - psi - beta)
        
        # Ensure omega is positive
        if omega <= 0:
            return 1e10
            
        # Initialize with unconditional values
        sigma2[0] = unconditional_variance
        
        # ARMA filtering and GARCH process
        for t in range(1, n):
            # AR component
            mu_t = phi * returns[t-1]
            
            # MA component (using previous error)
            if t > 1:
                mu_t += theta * eps[t-1]
                
            # Current error
            eps[t] = returns[t] - mu_t
            
            # GARCH variance update
            sigma2[t] = omega + psi * sigma2[t-1] + beta * eps[t-1]**2
            
        # Calculate log-likelihood with Student-T distribution
        v = 0.5 * (nu + 1)
        log_likelihood = n * (np.log(gamma(v)) - np.log(gamma(0.5 * nu)) - 0.5 * np.log(np.pi * (nu - 2)))
        
        for t in range(1, n):
            log_likelihood -= 0.5 * np.log(sigma2[t])
            log_likelihood -= v * np.log(1 + (eps[t]**2) / (sigma2[t] * (nu - 2)))
            
        return -log_likelihood
    
    def fit(self, returns):
        """
        Fit the ARMA-GARCH model using rolling window approach.
        
        Parameters:
            returns (array_like): Return series
            
        Returns:
            self: The fitted model instance
        """
        n = len(returns)
        result_params = []
        fitted_variances = []
        fitted_residuals = []
        fitted_innovations = []
        
        for i in range(self.window_size, n):
            window = returns[i-self.window_size:i]
            
            # Initial parameter guess
            initial_params = [0.1, 0.1, 0.3, 0.6, 8.0]  # [phi, theta, psi, beta, nu]
            
            # Parameter bounds
            bounds = [
                (-0.99, 0.99),  # phi
                (-0.99, 0.99),  # theta
                (0.01, 0.3),    # psi (GARCH)
                (0.01, 0.97),   # beta (ARCH)
                (2.1, 30.0)     # nu (degrees of freedom)
            ]
            
            # Constraint: psi + beta < 1 (variance stationarity)
            constraints = ({'type': 'ineq', 'fun': lambda x: 1 - x[2] - x[3]})
            
            # Optimize
            result = minimize(
                self.log_likelihood,
                initial_params,
                args=(window,),
                bounds=bounds,
                constraints=constraints,
                method='COBYLA'
            )
            
            # Store parameters
            result_params.append(result.x)
            
            # Calculate fitted values and residuals for this window
            phi, theta, psi, beta, nu = result.x
            
            # Calculate unconditional variance
            unconditional_variance = np.var(window)
            omega = unconditional_variance * (1 - psi - beta)
            
            # One-step forecast
            eps_last = 0
            sigma2_last = unconditional_variance
            
            if len(fitted_residuals) > 0:
                eps_last = fitted_residuals[-1]
                sigma2_last = fitted_variances[-1]
                
            # Forecast for next period
            mu_next = phi * window[-1] + theta * eps_last
            sigma2_next = omega + psi * sigma2_last + beta * eps_last**2
            
            # Store forecasts
            fitted_variances.append(sigma2_next)
            
            # Calculate residual when next return is available
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
        """
        Forecast Value-at-Risk.
        
        Parameters:
            alpha (float, optional): Confidence level (e.g., 0.05 for 95% VaR)
                If None, uses the instance alpha value.
            
        Returns:
            array_like: VaR forecasts
        """
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
    
    def forecast_cvar(self, alpha=None):
        """
        Forecast Conditional Value-at-Risk (Expected Shortfall).
        
        Parameters:
            alpha (float, optional): Confidence level (e.g., 0.05 for 95% CVaR)
                If None, uses the instance alpha value.
            
        Returns:
            array_like: CVaR forecasts
        """
        if alpha is None:
            alpha = self.alpha
            
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")
        
        cvar_forecasts = []
        
        for i, param_set in enumerate(self.params):
            _, _, _, _, nu = param_set
            sigma = np.sqrt(self.fitted_variances[i])
            
            # Calculate Student-T quantile
            t_quantile = t.ppf(alpha, nu)
            
            # Calculate PDF at quantile
            pdf_val = t.pdf(t_quantile, nu)
            
            # Calculate CVaR for Student-T
            # For Student's t: E[X|X<q] = µ - σ * (df * pdf(q) / ((df-1) * alpha))
            scale_factor = nu * pdf_val / ((nu - 1) * alpha)
            cvar_forecast = -sigma * scale_factor
            
            cvar_forecasts.append(cvar_forecast)
            
        return np.array(cvar_forecasts)
