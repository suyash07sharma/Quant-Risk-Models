import unittest
import numpy as np
from models.arma_garch import ARMAGARCH
from analysis.backtesting import var_exceedance_test, independence_test, kolmogorov_smirnov_test

class TestARMAGARCH(unittest.TestCase):
    def setUp(self):
        # Generate synthetic returns data with volatility clustering
        np.random.seed(42)
        n = 1000
        
        # GARCH process parameters
        omega = 0.00001
        alpha = 0.1
        beta = 0.8
        
        # ARMA parameters
        phi = 0.1
        theta = -0.1
        
        # Initialize arrays
        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        epsilon = np.zeros(n)
        
        # Initial variance
        sigma2[0] = 0.0001
        
        # Generate data
        for t in range(1, n):
            # GARCH variance
            sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
            
            # Random innovation
            z = np.random.standard_t(df=8)
            epsilon[t] = np.sqrt(sigma2[t]) * z
            
            # ARMA returns
            if t > 1:
                returns[t] = phi * returns[t-1] + epsilon[t] + theta * epsilon[t-1]
            else:
                returns[t] = epsilon[t]
        
        self.returns = returns
        self.model = ARMAGARCH(p=1, q=1, alpha=0.05, window_size=500)
        
    def test_model_fit(self):
        """Test if the model can be fitted without errors."""
        self.model.fit(self.returns)
        self.assertIsNotNone(self.model.params)
        self.assertEqual(len(self.model.params), len(self.returns) - self.model.window_size)
        
    def test_var_forecast(self):
        """Test VaR forecasting functionality."""
        self.model.fit(self.returns)
        var_95 = self.model.forecast_var(alpha=0.05)
        var_99 = self.model.forecast_var(alpha=0.01)
        
        # Check if VaRs are calculated
        self.assertEqual(len(var_95), len(self.model.params))
        self.assertEqual(len(var_99), len(self.model.params))
        
        # Check if 99% VaR is more extreme than 95% VaR
        self.assertTrue(np.all(var_99 < var_95))
        
    def test_backtesting(self):
        """Test backtesting functionality."""
        self.model.fit(self.returns)
        var_95 = self.model.forecast_var(alpha=0.05)
        test_returns = self.returns[self.model.window_size:self.model.window_size+len(var_95)]
        
        # Run backtests
        var_results = var_exceedance_test(test_returns, var_95, alpha=0.05)
        
        # Check if results contain expected keys
        self.assertIn('exceedances', var_results)
        self.assertIn('expected', var_results)
        self.assertIn('p_value', var_results)
        self.assertIn('result', var_results)
        
    def test_parameter_bounds(self):
        """Test if fitted parameters are within bounds."""
        self.model.fit(self.returns)
        
        # Check each parameter set
        for params in self.model.params:
            phi, theta, psi, beta, nu = params
            
            # Check bounds
            self.assertTrue(-0.99 <= phi <= 0.99)
            self.assertTrue(-0.99 <= theta <= 0.99)
            self.assertTrue(0.01 <= psi <= 0.3)
            self.assertTrue(0.01 <= beta <= 0.97)
            self.assertTrue(psi + beta < 1)
            self.assertTrue(nu > 2)

if __name__ == '__main__':
    unittest.main()
