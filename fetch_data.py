import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import time

# Import configuration
try:
    from config import *
except ImportError:
    print("Warning: config.py not found. Please create config.py with your API keys.")
    print("Using default values...")
    FINNHUB_API_KEY = "your_finnhub_api_key_here"
    DATA_START_DATE = "2018-01-01"
    DATA_END_DATE = "2025-01-01"
    NEWS_DAYS_BACK = 30
    MAX_NEWS_ARTICLES_PER_COMPANY = 10

# List of renewable energy companies and their tickers
companies = {
    'Orsted': 'ORSTED.CO',
    'Vestas': 'VWS.CO',
    'NextEra Energy': 'NEE',
    'Iberdrola': 'IBE.MC',
    'Enel': 'ENEL.MI',
    'Brookfield Renewable': 'BEP',
    'EDP Renovaveis': 'EDPR.LS',
    'Siemens Gamesa': 'SGRE.MC',
    'Plug Power': 'PLUG',
    'First Solar': 'FSLR'
}

# Finnhub API configuration
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'

def get_finnhub_data(endpoint, params=None):
    """Helper function to make Finnhub API requests"""
    if params is None:
        params = {}
    
    params['token'] = FINNHUB_API_KEY
    response = requests.get(f"{FINNHUB_BASE_URL}/{endpoint}", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data from {endpoint}: {response.status_code}")
        return None

# Create directory for saving raw data
os.makedirs('data/raw', exist_ok=True)

# Download historical stock data from Yahoo Finance
print("=== Downloading stock data from Yahoo Finance ===")
for name, ticker in companies.items():
    print(f"Downloading {name} ({ticker})...")
    data = yf.download(ticker, start=DATA_START_DATE, end=DATA_END_DATE)
    data.to_csv(f'data/raw/{ticker}_stock_data.csv')

print("Yahoo Finance download complete.")

# Fetch additional data from Finnhub
print("\n=== Fetching additional data from Finnhub ===")

# Check if API key is configured
if FINNHUB_API_KEY == 'your_finnhub_api_key_here':
    print("Warning: Please set your FINNHUB_API_KEY environment variable or update the script with your API key.")
    print("You can get a free API key from https://finnhub.io/")
    print("Skipping Finnhub data fetch...")
else:
    # Fetch company profiles
    print("Fetching company profiles...")
    company_profiles = {}
    
    for name, ticker in companies.items():
        print(f"Fetching profile for {name} ({ticker})...")
        
        # Get company profile
        profile = get_finnhub_data('stock/profile2', {'symbol': ticker})
        if profile:
            company_profiles[ticker] = profile
        
        # Add delay to respect API rate limits
        time.sleep(0.1)
    
    # Save company profiles
    if company_profiles:
        profiles_df = pd.DataFrame.from_dict(company_profiles, orient='index')
        profiles_df.to_csv('data/raw/finnhub_company_profiles.csv')
        print(f"Saved company profiles for {len(company_profiles)} companies")
    
    # Fetch financial metrics for US companies (Finnhub has better coverage for US stocks)
    print("\nFetching financial metrics...")
    financial_metrics = {}
    
    # Focus on US companies for financial metrics
    us_companies = {name: ticker for name, ticker in companies.items() 
                   if not any(ext in ticker for ext in ['.CO', '.MC', '.LS', '.MI'])}
    
    for name, ticker in us_companies.items():
        print(f"Fetching financial metrics for {name} ({ticker})...")
        
        # Get financial metrics
        metrics = get_finnhub_data('quote', {'symbol': ticker})
        if metrics:
            financial_metrics[ticker] = metrics
        
        # Add delay to respect API rate limits
        time.sleep(0.1)
    
    # Save financial metrics
    if financial_metrics:
        metrics_df = pd.DataFrame.from_dict(financial_metrics, orient='index')
        metrics_df.to_csv('data/raw/finnhub_financial_metrics.csv')
        print(f"Saved financial metrics for {len(financial_metrics)} companies")
    
    # Fetch recent news for the companies
    print("\nFetching recent news...")
    all_news = []
    
    for name, ticker in companies.items():
        print(f"Fetching news for {name} ({ticker})...")
        
        # Get news from the configured number of days back
        end_date = datetime.now()
        start_date = end_date - timedelta(days=NEWS_DAYS_BACK)
        
        news = get_finnhub_data('company-news', {
            'symbol': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        })
        
        if news:
            for article in news[:MAX_NEWS_ARTICLES_PER_COMPANY]:  # Limit to configured number of articles
                article['company'] = name
                article['ticker'] = ticker
                all_news.append(article)
        
        # Add delay to respect API rate limits
        time.sleep(0.1)
    
    # Save news data
    if all_news:
        news_df = pd.DataFrame(all_news)
        news_df.to_csv('data/raw/finnhub_news.csv', index=False)
        print(f"Saved {len(all_news)} news articles")

print("\nData fetch complete!")


