"""
Enhanced Data Fetching Module for Renewable Energy Investment Analysis
Loads data into database and includes additional financial metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import logging
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

from config import *
from database import DatabaseManager, create_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataFetcher:
    """Enhanced data fetcher with database integration"""
    
    def __init__(self):
        """Initialize the data fetcher"""
        self.db_manager = DatabaseManager()
        self.companies = {
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
        
    def fetch_stock_data(self, ticker, start_date=None, end_date=None):
        """Fetch comprehensive stock data"""
        try:
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            hist = stock.history(start=start_date or DATA_START_DATE, 
                               end=end_date or DATA_END_DATE)
            
            # Fetch additional info
            info = stock.info
            
            # Fetch financial statements
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
            except:
                income_stmt = None
                balance_sheet = None
                cash_flow = None
            
            return {
                'history': hist,
                'info': info,
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_financial_metrics(self, stock_data):
        """Calculate comprehensive financial metrics"""
        if not stock_data or not stock_data['income_stmt'] is not None:
            return {}
        
        try:
            income_stmt = stock_data['income_stmt']
            balance_sheet = stock_data['balance_sheet']
            cash_flow = stock_data['cash_flow']
            info = stock_data['info']
            
            # Get most recent data
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]  # Most recent year
                prev_income = income_stmt.iloc[:, 1] if income_stmt.shape[1] > 1 else None
            else:
                latest_income = None
                prev_income = None
            
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
            else:
                latest_balance = None
            
            metrics = {}
            
            # Basic ratios from yfinance info
            metrics['PE_Ratio'] = info.get('trailingPE')
            metrics['PB_Ratio'] = info.get('priceToBook')
            metrics['PS_Ratio'] = info.get('priceToSalesTrailing12Months')
            metrics['Debt_to_Equity'] = info.get('debtToEquity')
            metrics['Current_Ratio'] = info.get('currentRatio')
            metrics['ROE'] = info.get('returnOnEquity')
            metrics['ROA'] = info.get('returnOnAssets')
            metrics['Profit_Margin'] = info.get('profitMargins')
            metrics['Revenue_Growth'] = info.get('revenueGrowth')
            metrics['Earnings_Growth'] = info.get('earningsGrowth')
            metrics['Dividend_Yield'] = info.get('dividendYield')
            metrics['Payout_Ratio'] = info.get('payoutRatio')
            
            # Calculate additional metrics if financial statements available
            if latest_income is not None and latest_balance is not None:
                try:
                    # Revenue growth
                    if prev_income is not None and 'Total Revenue' in latest_income and 'Total Revenue' in prev_income:
                        current_revenue = latest_income['Total Revenue']
                        prev_revenue = prev_income['Total Revenue']
                        if prev_revenue and prev_revenue != 0:
                            metrics['Revenue_Growth_Calculated'] = (current_revenue - prev_revenue) / abs(prev_revenue)
                    
                    # Earnings growth
                    if prev_income is not None and 'Net Income' in latest_income and 'Net Income' in prev_income:
                        current_earnings = latest_income['Net Income']
                        prev_earnings = prev_income['Net Income']
                        if prev_earnings and prev_earnings != 0:
                            metrics['Earnings_Growth_Calculated'] = (current_earnings - prev_earnings) / abs(prev_earnings)
                    
                    # ROE calculation
                    if 'Net Income' in latest_income and 'Total Stockholder Equity' in latest_balance:
                        net_income = latest_income['Net Income']
                        equity = latest_balance['Total Stockholder Equity']
                        if equity and equity != 0:
                            metrics['ROE_Calculated'] = net_income / equity
                    
                    # ROA calculation
                    if 'Net Income' in latest_income and 'Total Assets' in latest_balance:
                        net_income = latest_income['Net Income']
                        assets = latest_balance['Total Assets']
                        if assets and assets != 0:
                            metrics['ROA_Calculated'] = net_income / assets
                    
                    # Profit margin calculation
                    if 'Net Income' in latest_income and 'Total Revenue' in latest_income:
                        net_income = latest_income['Net Income']
                        revenue = latest_income['Total Revenue']
                        if revenue and revenue != 0:
                            metrics['Profit_Margin_Calculated'] = net_income / revenue
                    
                except Exception as e:
                    logger.warning(f"Error calculating financial metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return {}
    
    def fetch_news_sentiment(self, ticker, days_back=30):
        """Fetch news and calculate sentiment scores"""
        if FINNHUB_API_KEY == 'your_finnhub_api_key_here':
            logger.warning("Finnhub API key not configured. Skipping news sentiment analysis.")
            return []
        
        try:
            # Fetch news from Finnhub
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{FINNHUB_BASE_URL}/company-news"
            params = {
                'symbol': ticker,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': FINNHUB_API_KEY
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                news_data = response.json()
                
                # Calculate sentiment for each article
                articles_with_sentiment = []
                for article in news_data[:MAX_NEWS_ARTICLES_PER_COMPANY]:
                    # Calculate sentiment using TextBlob
                    headline_sentiment = TextBlob(article.get('headline', '')).sentiment.polarity
                    summary_sentiment = TextBlob(article.get('summary', '')).sentiment.polarity
                    
                    # Average sentiment
                    avg_sentiment = (headline_sentiment + summary_sentiment) / 2
                    
                    article['sentiment_score'] = avg_sentiment
                    article['relevance_score'] = 0.8  # Placeholder relevance score
                    articles_with_sentiment.append(article)
                
                return articles_with_sentiment
            else:
                logger.error(f"Failed to fetch news for {ticker}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def load_data_to_database(self):
        """Load all data into the database"""
        logger.info("Starting data loading process...")
        
        # Create database and tables if they don't exist
        if not create_database():
            logger.error("Failed to create database")
            return False
        
        # Connect to database
        if not self.db_manager.connect():
            logger.error("Failed to connect to database")
            return False
        
        try:
            # Load companies
            logger.info("Loading company information...")
            self.db_manager.load_companies(self.companies)
            
            # Fetch and load data for each company
            for name, ticker in self.companies.items():
                logger.info(f"Processing {name} ({ticker})...")
                
                # Fetch stock data
                stock_data = self.fetch_stock_data(ticker)
                if stock_data and not stock_data['history'].empty:
                    # Load stock prices
                    self.db_manager.load_stock_prices(ticker, stock_data['history'])
                    
                    # Calculate and load financial metrics
                    financial_metrics = self.calculate_financial_metrics(stock_data)
                    if financial_metrics:
                        # Convert to DataFrame for loading
                        metrics_df = pd.DataFrame([financial_metrics], index=[datetime.now().date()])
                        self.db_manager.load_financial_metrics(ticker, metrics_df)
                    
                    # Fetch and load news sentiment
                    news_articles = self.fetch_news_sentiment(ticker)
                    if news_articles:
                        # Load news articles (this would need to be implemented in DatabaseManager)
                        logger.info(f"Fetched {len(news_articles)} news articles for {ticker}")
                
                # Add delay to respect API rate limits
                time.sleep(1)
            
            logger.info("Data loading completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def generate_data_summary(self):
        """Generate summary of loaded data"""
        if not self.db_manager.connect():
            logger.error("Failed to connect to database")
            return
        
        try:
            # Get company summary
            summary = self.db_manager.get_company_summary()
            
            print("\n" + "="*80)
            print("DATA LOADING SUMMARY")
            print("="*80)
            
            if not summary.empty:
                print(f"\nLoaded data for {len(summary)} companies:")
                for _, row in summary.iterrows():
                    print(f"  • {row['company_name']} ({row['ticker']})")
                    if row['current_price']:
                        print(f"    Current Price: ${row['current_price']:.2f}")
                    if row['pe_ratio']:
                        print(f"    P/E Ratio: {row['pe_ratio']:.2f}")
                    if row['roe']:
                        print(f"    ROE: {row['roe']:.2%}")
                    print()
            else:
                print("No data found in database")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        finally:
            self.db_manager.disconnect()

def main():
    """Main function to run the enhanced data fetching"""
    fetcher = EnhancedDataFetcher()
    
    print("Enhanced Renewable Energy Data Fetcher")
    print("="*50)
    
    # Load data to database
    success = fetcher.load_data_to_database()
    
    if success:
        # Generate summary
        fetcher.generate_data_summary()
        print("\nData fetching completed successfully!")
    else:
        print("\nData fetching failed. Please check the logs for details.")

if __name__ == "__main__":
    main() 