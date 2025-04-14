import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def download_data(tickers, start_date=None, end_date=None, period='5y', interval='1d'):
    """
    Download historical price data for a list of tickers.
    
    Parameters:
        tickers (list): List of ticker symbols
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        period (str, optional): Period to download (e.g., '5y' for 5 years)
        interval (str, optional): Data interval (e.g., '1d' for daily)
        
    Returns:
        dict: Dictionary mapping tickers to DataFrames
    """
    data_dict = {}
    
    # If dates are provided, use them instead of period
    if start_date and end_date:
        period = None
    
    for ticker in tickers:
        try:
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                period=period,
                interval=interval,
                progress=False
            )
            
            # Check if data is valid
            if data.empty:
                print(f"No data found for {ticker}")
                continue
                
            # Store data
            data_dict[ticker] = data
            
            print(f"Downloaded {len(data)} rows for {ticker}")
            
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return data_dict

def calculate_returns(data_dict, return_type='log'):
    """
    Calculate returns from price data.
    
    Parameters:
        data_dict (dict): Dictionary mapping tickers to DataFrames
        return_type (str): Type of return ('log' or 'simple')
        
    Returns:
        dict: Dictionary mapping tickers to return DataFrames
    """
    returns_dict = {}
    
    for ticker, df in data_dict.items():
        # Create a copy
        returns_df = df.copy()
        
        # Calculate returns
        if return_type == 'log':
            returns_df['return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        else:  # simple returns
            returns_df['return'] = df['Adj Close'] / df['Adj Close'].shift(1) - 1
            
        # Drop NaN values
        returns_df = returns_df.dropna(subset=['return'])
        
        # Store in dictionary
        returns_dict[ticker] = returns_df
    
    return returns_dict

def load_sample_data():
    """
    Load a small sample dataset for testing and demonstration.
    
    Returns:
        dict: Dictionary with sample data
    """
    # Standard tickers for analysis
    tickers = ['SPY', 'IYR', 'GLD', 'USO', 'TLT']
    
    # Calculate end date (today) and start date (5 years ago)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    
    # Format dates
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data
    data_dict = download_data(tickers, start_date=start_str, end_date=end_str)
    
    # Calculate returns
    returns_dict = calculate_returns(data_dict)
    
    return {
        'tickers': tickers,
        'price_data': data_dict,
        'returns_data': returns_dict
    }
