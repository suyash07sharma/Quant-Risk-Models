import numpy as np
from scipy.stats import binom, norm, kstest

def var_exceedance_test(returns, var_values, alpha=0.05):
    """
    Perform VaR exceedance test (Kupiec test).
    
    Parameters:
        returns (array_like): Actual returns
        var_values (array_like): Predicted VaR values
        alpha (float): VaR significance level
        
    Returns:
        dict: Test results including exceedances, p-value, and test verdict
    """
    # Count exceedances
    exceedances = sum(returns < var_values)
    n = len(returns)
    expected = n * alpha
    
    # Calculate p-value using binomial test
    p_value = 2 * min(
        binom.cdf(exceedances, n, alpha),
        binom.sf(exceedances - 1, n, alpha)
    )
    
    # Calculate confidence bounds
    lower_bound = binom.ppf(0.025, n, alpha)
    upper_bound = binom.ppf(0.975, n, alpha)
    
    return {
        'exceedances': exceedances,
        'expected': expected,
        'sample_size': n,
        'rate': exceedances / n,
        'expected_rate': alpha,
        'p_value': p_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'result': 'Accepted' if p_value > 0.05 else 'Rejected',
        'direction': 'Over-prediction' if exceedances < expected else 'Under-prediction',
    }

def independence_test(returns, var_values):
    """
    Perform the Christoffersen test for independence of VaR exceedances.
    
    Parameters:
        returns (array_like): Actual returns
        var_values (array_like): Predicted VaR values
        
    Returns:
        dict: Test results
    """
    # Create exceedance indicator series
    indicators = (returns < var_values).astype(int)
    n = len(indicators)
    
    # Count transitions
    n00 = n01 = n10 = n11 = 0
    
    for i in range(n-1):
        if indicators[i] == 0 and indicators[i+1] == 0:
            n00 += 1
        elif indicators[i] == 0 and indicators[i+1] == 1:
            n01 += 1
        elif indicators[i] == 1 and indicators[i+1] == 0:
            n10 += 1
        else:  # indicators[i] == 1 and indicators[i+1] == 1
            n11 += 1
    
    # Calculate transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    
    # Calculate the overall probability of exceedance
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Calculate likelihood values
    L_ind = ((1 - p01) ** n00) * (p01 ** n01) * ((1 - p11) ** n10) * (p11 ** n11)
    L_0 = ((1 - p) ** (n00 + n10)) * (p ** (n01 + n11))
    
    # Calculate test statistic
    LR_ind = -2 * np.log(L_0 / L_ind) if L_ind > 0 else 0
    
    # p-value (chi-squared with 1 degree of freedom)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(LR_ind, 1)
    
    return {
        'n00': n00,
        'n01': n01,
        'n10': n10,
        'n11': n11,
        'p01': p01,
        'p11': p11,
        'LR_statistic': LR_ind,
        'p_value': p_value,
        'result': 'Accepted' if p_value > 0.05 else 'Rejected'
    }

def christoffersen_test(returns, var_values, alpha=0.05):
    """
    Perform the combined Christoffersen test (coverage and independence).
    
    Parameters:
        returns (array_like): Actual returns
        var_values (array_like): Predicted VaR values
        alpha (float): VaR significance level
        
    Returns:
        dict: Test results
    """
    # Get individual test results
    kupiec_results = var_exceedance_test(returns, var_values, alpha)
    independence_results = independence_test(returns, var_values)
    
    # Combined test statistic
    LR_cc = -2 * kupiec_results['p_value'] + -2 * independence_results['p_value']
    
    # p-value (chi-squared with 2 degrees of freedom)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(LR_cc, 2)
    
    return {
        'kupiec_test': kupiec_results,
        'independence_test': independence_results,
        'LR_statistic': LR_cc,
        'p_value': p_value,
        'result': 'Accepted' if p_value > 0.05 else 'Rejected'
    }

def kolmogorov_smirnov_test(model_quantiles):
    """
    Perform a Kolmogorov-Smirnov test to check if model quantiles follow a uniform distribution.
    
    Parameters:
        model_quantiles (array_like): Model-transformed quantiles
        
    Returns:
        dict: Test results
    """
    # Perform KS test against uniform distribution
    ks_stat, p_value = kstest(model_quantiles, 'uniform')
    
    return {
        'KS_statistic': ks_stat,
        'p_value': p_value,
        'result': 'Accepted' if p_value > 0.05 else 'Rejected'
    }
