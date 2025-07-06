"""
Enhanced Data Fetching Module for Renewable Energy Investment Analysis
Unified data collection with export to both CSV (PowerBI) and database (main analysis)
"""

import sys
import traceback
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import validation with clear error messages
try:
    import yfinance as yf
except ImportError as e:
    print("❌ ERROR: Required package 'yfinance' not found!")
    print(f"   Details: {e}")
    print("   Solution: Run 'pip install yfinance'")
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print("❌ ERROR: Required package 'pandas' not found!")
    print(f"   Details: {e}")
    print("   Solution: Run 'pip install pandas'")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    print("❌ ERROR: Required package 'numpy' not found!")
    print(f"   Details: {e}")
    print("   Solution: Run 'pip install numpy'")
    sys.exit(1)

try:
    import requests
except ImportError as e:
    print("❌ ERROR: Required package 'requests' not found!")
    print(f"   Details: {e}")
    print("   Solution: Run 'pip install requests'")
    sys.exit(1)

try:
    from textblob import TextBlob
except ImportError as e:
    print("⚠️  WARNING: Package 'textblob' not found. News sentiment analysis will be skipped.")
    print(f"   Details: {e}")
    print("   Solution: Run 'pip install textblob'")
    TextBlob = None

# Import local modules with error handling
try:
    from config import *
except ImportError as e:
    print("❌ ERROR: Configuration file 'config.py' not found or invalid!")
    print(f"   Details: {e}")
    print("   Solution: Ensure config.py exists and contains required variables")
    sys.exit(1)

try:
    from database import DatabaseManager, create_database
except ImportError as e:
    print("❌ ERROR: Database module 'database.py' not found or invalid!")
    print(f"   Details: {e}")
    print("   Solution: Ensure database.py exists and is properly formatted")
    sys.exit(1)

try:
    from investment_analysis import InvestmentAnalyzer
except ImportError as e:
    print("❌ ERROR: Investment analysis module 'investment_analysis.py' not found or invalid!")
    print(f"   Details: {e}")
    print("   Solution: Ensure investment_analysis.py exists and is properly formatted")
    sys.exit(1)

import logging

# Set up logging to show INFO messages in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/fetch_data.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_configuration():
    """Validate that all required configuration variables are present"""
    print("🔍 Validating configuration...")
    
    required_vars = [
        'FINNHUB_API_KEY', 'FINNHUB_BASE_URL', 'DATA_START_DATE', 
        'DATA_END_DATE', 'RISK_FREE_RATE', 'MARKET_RETURN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not globals().get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ ERROR: Missing required configuration variables: {', '.join(missing_vars)}")
        print("   Solution: Check your config.py file and ensure all variables are defined")
        return False
    
    # Validate date formats
    try:
        datetime.strptime(DATA_START_DATE, '%Y-%m-%d')
        datetime.strptime(DATA_END_DATE, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ ERROR: Invalid date format in configuration: {e}")
        print("   Solution: Ensure dates are in YYYY-MM-DD format")
        return False
    
    print("✅ Configuration validation passed")
    return True

class EnhancedDataFetcher:
    """Enhanced data fetcher with unified collection and export"""
    
    def __init__(self):
        """Initialize the data fetcher"""
        print("🚀 Initializing Enhanced Data Fetcher...")
        
        try:
            self.db_manager = DatabaseManager()
            self.analyzer = InvestmentAnalyzer()
            
            # Create output directories
            directories = ['powerbi/data', 'analysis', 'reports', 'charts', 'logs']
            for directory in directories:
                try:
                    os.makedirs(directory, exist_ok=True)
                    print(f"✅ Created directory: {directory}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not create directory {directory}: {e}")
            
            # Filter out delisted companies (same as powerbi_export.py)
            companies_dict = {
                'Orsted': 'ORSTED.CO',
                'Vestas': 'VWS.CO',
                'NextEra Energy': 'NEE',
                'Iberdrola': 'IBE.MC',
                'Enel': 'ENEL.MI',
                'Brookfield Renewable': 'BEP',
                'EDP Renovaveis': 'EDPR.LS',
                'Plug Power': 'PLUG',
                'First Solar': 'FSLR'
            }
            self.companies = {}
            for name, ticker in companies_dict.items():
                if ticker != 'SGRE.MC':  # Remove delisted company
                    self.companies[name] = ticker
            
            print(f"✅ Loaded {len(self.companies)} companies for analysis")
            
            # Store collected data for reuse
            self.collected_data = {}
            self.company_summaries = {}
            self.stock_prices = {}
            self.risk_metrics = {}
            
            print("✅ Enhanced Data Fetcher initialized successfully")
            
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Enhanced Data Fetcher: {e}")
            print(f"   Stack trace: {traceback.format_exc()}")
            raise
        
    def fetch_stock_data(self, ticker, start_date=None, end_date=None):
        """Fetch comprehensive stock data"""
        try:
            print(f"📊 Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            hist = stock.history(start=start_date or DATA_START_DATE, 
                               end=end_date or DATA_END_DATE)
            
            if hist.empty:
                print(f"⚠️  Warning: No historical data found for {ticker}")
                return None
            
            # Fetch additional info
            info = stock.info
            
            # Fetch financial statements
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
                print(f"✅ Financial statements fetched for {ticker}")
            except Exception as e:
                print(f"⚠️  Warning: Could not fetch financial statements for {ticker}: {e}")
                income_stmt = None
                balance_sheet = None
                cash_flow = None
            
            print(f"✅ Successfully fetched data for {ticker}")
            return {
                'history': hist,
                'info': info,
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            print(f"❌ ERROR: Failed to fetch data for {ticker}: {e}")
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_financial_metrics(self, stock_data):
        """Calculate comprehensive financial metrics"""
        if not stock_data or stock_data['income_stmt'] is None:
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
                    print(f"⚠️  Warning: Error calculating financial metrics: {e}")
                    logger.warning(f"Error calculating financial metrics: {e}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ ERROR: Failed to calculate financial metrics: {e}")
            logger.error(f"Error calculating financial metrics: {e}")
            return {}
    
    def fetch_news_sentiment(self, ticker, days_back=30):
        """Fetch news and calculate sentiment scores"""
        if FINNHUB_API_KEY == 'your_finnhub_api_key_here':
            print("⚠️  Warning: Finnhub API key not configured. Skipping news sentiment analysis.")
            logger.warning("Finnhub API key not configured. Skipping news sentiment analysis.")
            return []
        
        if TextBlob is None:
            print("⚠️  Warning: TextBlob not available. Skipping news sentiment analysis.")
            return []
        
        try:
            print(f"📰 Fetching news for {ticker}...")
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
                
                print(f"✅ Fetched {len(articles_with_sentiment)} news articles for {ticker}")
                return articles_with_sentiment
            else:
                print(f"❌ ERROR: Failed to fetch news for {ticker}: HTTP {response.status_code}")
                logger.error(f"Failed to fetch news for {ticker}: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ ERROR: Failed to fetch news for {ticker}: {e}")
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def collect_all_data(self):
        """Collect all data once for both PowerBI export and database storage"""
        print("\n📈 Starting unified data collection...")
        
        try:
            # Step 1: Collect stock data and company information
            print("Step 1: Collecting stock data and company information...")
            successful_companies = 0
            
            for name, ticker in self.companies.items():
                print(f"\n📊 Processing {name} ({ticker})...")
                
                try:
                    # Fetch comprehensive data
                    stock_data = self.fetch_stock_data(ticker)
                    
                    if stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
                        # Store stock prices
                        self.stock_prices[ticker] = stock_data['history']
                        
                        # Calculate financial metrics
                        financial_metrics = self.calculate_financial_metrics(stock_data)
                        
                        # Create company summary
                        current_price = stock_data['history']['Close'].iloc[-1]
                        info = stock_data['info']
                        
                        self.company_summaries[ticker] = {
                            'company_id': len(self.company_summaries) + 1,
                            'ticker': ticker,
                            'company_name': name,
                            'sector': 'Renewable Energy',
                            'industry': 'Renewable Energy',
                            'market_cap': info.get('marketCap'),
                            'current_price': current_price,
                            'last_price_date': datetime.now().date(),
                            'pe_ratio': info.get('trailingPE'),
                            'pb_ratio': info.get('priceToBook'),
                            'roe': info.get('returnOnEquity'),
                            'debt_to_equity': info.get('debtToEquity'),
                            'volatility': None,  # Will be calculated later
                            'beta': info.get('beta'),
                            'sharpe_ratio': None,  # Will be calculated later
                            **financial_metrics  # Add all calculated financial metrics
                        }
                        
                        # Calculate risk metrics
                        returns = stock_data['history']['Close'].pct_change().dropna()
                        if len(returns) > 30:  # Need sufficient data
                            try:
                                risk_metrics = self.analyzer.risk_analysis(returns)
                                self.risk_metrics[ticker] = risk_metrics
                                
                                # Update company summary with risk metrics
                                self.company_summaries[ticker]['volatility'] = risk_metrics['Volatility']
                                self.company_summaries[ticker]['sharpe_ratio'] = risk_metrics['Sharpe_Ratio']
                                print(f"✅ Risk metrics calculated for {ticker}")
                            except Exception as e:
                                print(f"⚠️  Warning: Could not calculate risk metrics for {ticker}: {e}")
                        
                        successful_companies += 1
                        print(f"✅ Successfully collected data for {ticker}")
                        
                    else:
                        print(f"⚠️  Warning: No valid data found for {ticker}")
                        
                except Exception as e:
                    print(f"❌ ERROR: Failed to collect data for {ticker}: {e}")
                    logger.error(f"Error collecting data for {ticker}: {e}")
                    continue
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
            
            print(f"\n✅ Data collection completed. Successfully processed {successful_companies}/{len(self.companies)} companies.")
            
            if successful_companies == 0:
                print("❌ ERROR: No companies were successfully processed!")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Failed during data collection: {e}")
            print(f"   Stack trace: {traceback.format_exc()}")
            logger.error(f"Error in unified data collection: {e}")
            return False
    
    def export_powerbi_data(self):
        """Export all data to CSV files for PowerBI"""
        print("\n📊 Exporting data to PowerBI CSV files...")
        
        success = True
        exported_files = []
        
        # 1. Export company summary
        if self.company_summaries:
            try:
                print("📋 Exporting company summary...")
                summary_df = pd.DataFrame(list(self.company_summaries.values()))
                
                # Add calculated fields
                summary_df['Market_Cap_Billions'] = summary_df['market_cap'] / 1e9 if 'market_cap' in summary_df.columns else None
                summary_df['PE_Category'] = pd.cut(summary_df['pe_ratio'], 
                                                  bins=[0, 15, 25, 50, float('inf')], 
                                                  labels=['Value', 'Fair Value', 'Growth', 'High Growth'])
                summary_df['Risk_Category'] = pd.cut(summary_df['volatility'], 
                                                    bins=[0, 0.2, 0.4, 0.6, float('inf')], 
                                                    labels=['Low', 'Medium', 'High', 'Very High'])
                
                output_path = 'powerbi/data/company_summary.csv'
                summary_df.to_csv(output_path, index=False)
                exported_files.append('company_summary.csv')
                print(f"✅ Company summary exported to {output_path}")
            except Exception as e:
                print(f"❌ ERROR: Failed to export company summary: {e}")
                logger.error(f"Error exporting company summary: {e}")
                success = False
        else:
            print("⚠️  Warning: No company summaries to export")
        
        # 2. Export stock prices
        if self.stock_prices:
            try:
                print("📈 Exporting stock prices...")
                all_prices = []
                
                for name, ticker in self.companies.items():
                    if ticker in self.stock_prices:
                        prices = self.stock_prices[ticker].copy()
                        prices['Ticker'] = ticker
                        prices['Company_Name'] = name
                        prices.reset_index(inplace=True)
                        # Ensure the index column (now Date) is properly named and converted
                        if 'Date' not in prices.columns:
                            date_col = prices.columns[0]  # First column should be the date
                            prices.rename(columns={date_col: 'Date'}, inplace=True)
                        # Ensure Date column is datetime type
                        prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce')
                        all_prices.append(prices)
                
                if all_prices:
                    combined_prices = pd.concat(all_prices, ignore_index=True)
                    # Ensure Date column is datetime type before using .dt
                    combined_prices['Date'] = pd.to_datetime(combined_prices['Date'], errors='coerce')
                    # Add calculated fields - now safe to use .dt accessor
                    combined_prices['Year'] = combined_prices['Date'].dt.year
                    combined_prices['Month'] = combined_prices['Date'].dt.month
                    combined_prices['Quarter'] = combined_prices['Date'].dt.quarter
                    combined_prices['Day_of_Week'] = combined_prices['Date'].dt.dayofweek
                    
                    # Calculate returns
                    combined_prices['Daily_Return'] = combined_prices.groupby('Ticker')['Close'].pct_change()
                    combined_prices['Cumulative_Return'] = combined_prices.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod())
                    
                    output_path = 'powerbi/data/stock_prices.csv'
                    combined_prices.to_csv(output_path, index=False)
                    exported_files.append('stock_prices.csv')
                    print(f"✅ Stock prices exported to {output_path}")
                else:
                    print("⚠️  Warning: No stock prices to export")
            except Exception as e:
                print(f"❌ ERROR: Failed to export stock prices: {e}")
                logger.error(f"Error exporting stock prices: {e}")
                success = False
        else:
            print("⚠️  Warning: No stock prices to export")
        
        # 3. Export risk metrics
        if self.risk_metrics:
            try:
                print("⚠️  Exporting risk metrics...")
                all_risk_data = []
                
                for name, ticker in self.companies.items():
                    if ticker in self.risk_metrics:
                        risk_metrics = self.risk_metrics[ticker].copy()
                        risk_metrics['Ticker'] = ticker
                        risk_metrics['Company_Name'] = name
                        risk_metrics['Analysis_Date'] = datetime.now().date()
                        all_risk_data.append(risk_metrics)
                
                if all_risk_data:
                    risk_df = pd.DataFrame(all_risk_data)
                    
                    # Add risk categories
                    risk_df['Risk_Level'] = pd.cut(risk_df['Volatility'], 
                                                 bins=[0, 0.2, 0.4, 0.6, float('inf')], 
                                                 labels=['Low', 'Medium', 'High', 'Very High'])
                    risk_df['Sharpe_Category'] = pd.cut(risk_df['Sharpe_Ratio'], 
                                                       bins=[float('-inf'), 0, 0.5, 1, float('inf')], 
                                                       labels=['Poor', 'Fair', 'Good', 'Excellent'])
                    
                    output_path = 'powerbi/data/risk_metrics.csv'
                    risk_df.to_csv(output_path, index=False)
                    exported_files.append('risk_metrics.csv')
                    print(f"✅ Risk metrics exported to {output_path}")
                else:
                    print("⚠️  Warning: No risk metrics to export")
            except Exception as e:
                print(f"❌ ERROR: Failed to export risk metrics: {e}")
                logger.error(f"Error exporting risk metrics: {e}")
                success = False
        else:
            print("⚠️  Warning: No risk metrics to export")
        
        # 4. Export portfolio analysis
        if len(self.stock_prices) >= 2:
            try:
                print("📊 Exporting portfolio analysis...")
                # Get returns data
                all_returns = {}
                for ticker, prices in self.stock_prices.items():
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
                
                returns_df = pd.DataFrame(all_returns)
                returns_df = returns_df.dropna()
                
                if not returns_df.empty:
                    portfolio_data = []
                    
                    for method in ['sharpe', 'min_variance', 'max_return', 'equal_weight']:
                        try:
                            if method == 'equal_weight':
                                # Equal weight portfolio
                                n_assets = len(returns_df.columns)
                                weights = np.array([1/n_assets] * n_assets)
                                expected_returns = returns_df.mean() * 252
                                cov_matrix = returns_df.cov() * 252
                                portfolio_return = np.sum(expected_returns * weights)
                                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                                sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
                                
                                result = {
                                    'weights': weights,
                                    'expected_return': portfolio_return,
                                    'volatility': portfolio_vol,
                                    'sharpe_ratio': sharpe_ratio
                                }
                            else:
                                result = self.analyzer.portfolio_optimization(returns_df, method)
                            
                            if result:
                                for i, ticker in enumerate(returns_df.columns):
                                    portfolio_data.append({
                                        'Optimization_Method': method.replace('_', ' ').title(),
                                        'Ticker': ticker,
                                        'Weight': result['weights'][i],
                                        'Expected_Return': result['expected_return'],
                                        'Portfolio_Volatility': result['volatility'],
                                        'Sharpe_Ratio': result['sharpe_ratio']
                                    })
                        except Exception as e:
                            print(f"⚠️  Warning: Could not calculate {method} portfolio: {e}")
                    
                    if portfolio_data:
                        portfolio_df = pd.DataFrame(portfolio_data)
                        output_path = 'powerbi/data/portfolio_analysis.csv'
                        portfolio_df.to_csv(output_path, index=False)
                        exported_files.append('portfolio_analysis.csv')
                        print(f"✅ Portfolio analysis exported to {output_path}")
                    else:
                        print("⚠️  Warning: No portfolio analysis data to export")
                else:
                    print("⚠️  Warning: Insufficient returns data for portfolio analysis")
            except Exception as e:
                print(f"❌ ERROR: Failed to export portfolio analysis: {e}")
                logger.error(f"Error exporting portfolio analysis: {e}")
                success = False
        else:
            print("⚠️  Warning: Need at least 2 companies for portfolio analysis")
        
        # 5. Export technical indicators
        if self.stock_prices:
            try:
                print("📊 Exporting technical indicators...")
                all_technical_data = []
                
                for name, ticker in self.companies.items():
                    if ticker in self.stock_prices:
                        try:
                            prices = self.stock_prices[ticker]
                            technical = self.analyzer.technical_analysis(prices)
                            
                            tech_df = pd.DataFrame({
                                'Date': prices.index,
                                'Ticker': ticker,
                                'Company_Name': name,
                                'Close_Price': prices['Close'],
                                'SMA_20': technical['SMA_20'],
                                'SMA_50': technical['SMA_50'],
                                'SMA_200': technical['SMA_200'],
                                'RSI': technical['RSI'],
                                'MACD': technical['MACD'],
                                'MACD_Signal': technical['MACD_Signal'],
                                'MACD_Histogram': technical['MACD_Histogram'],
                                'BB_Upper': technical['BB_Upper'],
                                'BB_Middle': technical['BB_Middle'],
                                'BB_Lower': technical['BB_Lower']
                            })
                            
                            # Add signal indicators
                            tech_df['Price_vs_SMA20'] = np.where(tech_df['Close_Price'] > tech_df['SMA_20'], 'Above', 'Below')
                            tech_df['Price_vs_SMA50'] = np.where(tech_df['Close_Price'] > tech_df['SMA_50'], 'Above', 'Below')
                            tech_df['RSI_Signal'] = np.where(tech_df['RSI'] > 70, 'Overbought', 
                                                           np.where(tech_df['RSI'] < 30, 'Oversold', 'Neutral'))
                            tech_df['MACD_Signal'] = np.where(tech_df['MACD'] > tech_df['MACD_Signal'], 'Bullish', 'Bearish')
                            
                            all_technical_data.append(tech_df)
                        except Exception as e:
                            print(f"⚠️  Warning: Could not calculate technical indicators for {ticker}: {e}")
                
                if all_technical_data:
                    combined_technical = pd.concat(all_technical_data, ignore_index=True)
                    output_path = 'powerbi/data/technical_indicators.csv'
                    combined_technical.to_csv(output_path, index=False)
                    exported_files.append('technical_indicators.csv')
                    print(f"✅ Technical indicators exported to {output_path}")
                else:
                    print("⚠️  Warning: No technical indicators to export")
            except Exception as e:
                print(f"❌ ERROR: Failed to export technical indicators: {e}")
                logger.error(f"Error exporting technical indicators: {e}")
                success = False
        else:
            print("⚠️  Warning: No stock prices for technical indicators")
        
        # 6. Export correlation matrix
        if len(self.stock_prices) >= 2:
            try:
                print("📊 Exporting correlation matrix...")
                all_returns = {}
                for ticker, prices in self.stock_prices.items():
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
                
                returns_df = pd.DataFrame(all_returns)
                returns_df = returns_df.dropna()
                
                if not returns_df.empty:
                    correlation_matrix = returns_df.corr()
                    
                    correlation_data = []
                    for i, ticker1 in enumerate(correlation_matrix.index):
                        for j, ticker2 in enumerate(correlation_matrix.columns):
                            correlation_data.append({
                                'Ticker1': ticker1,
                                'Ticker2': ticker2,
                                'Correlation': correlation_matrix.iloc[i, j],
                                'Correlation_Abs': abs(correlation_matrix.iloc[i, j]),
                                'Correlation_Category': 'High' if abs(correlation_matrix.iloc[i, j]) > 0.7 else 
                                                      'Medium' if abs(correlation_matrix.iloc[i, j]) > 0.3 else 'Low'
                            })
                    
                    if correlation_data:
                        correlation_df = pd.DataFrame(correlation_data)
                        output_path = 'powerbi/data/correlation_matrix.csv'
                        correlation_df.to_csv(output_path, index=False)
                        exported_files.append('correlation_matrix.csv')
                        print(f"✅ Correlation matrix exported to {output_path}")
                    else:
                        print("⚠️  Warning: No correlation data to export")
                else:
                    print("⚠️  Warning: Insufficient returns data for correlation matrix")
            except Exception as e:
                print(f"❌ ERROR: Failed to export correlation matrix: {e}")
                logger.error(f"Error exporting correlation matrix: {e}")
                success = False
        else:
            print("⚠️  Warning: Need at least 2 companies for correlation matrix")
        
        if success and exported_files:
            print(f"\n✅ PowerBI export completed successfully!")
            print(f"📁 Exported {len(exported_files)} files to 'powerbi/data/' directory:")
            for file in exported_files:
                print(f"   • {file}")
        elif not exported_files:
            print("⚠️  Warning: No files were exported")
            success = False
        else:
            print("⚠️  Warning: Some exports failed")
        
        return success
    
    def load_data_to_database(self):
        """Load collected data into database for main analysis"""
        print("\n🗄️  Loading data to database...")
        
        try:
            # Create database and tables
            print("🔧 Creating database and tables...")
            if not create_database():
                print("⚠️  Warning: Failed to create database, continuing without database")
                logger.warning("Failed to create database, continuing without database")
                return True
            
            # Connect to database
            print("🔌 Connecting to database...")
            if not self.db_manager.connect():
                print("⚠️  Warning: Failed to connect to database, continuing without database")
                logger.warning("Failed to connect to database, continuing without database")
                return True
            
            # Load company summaries
            print("📋 Loading company summaries...")
            summary_count = 0
            for ticker, summary in self.company_summaries.items():
                try:
                    self.db_manager.insert_company_summary(summary)
                    summary_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Error inserting company summary for {ticker}: {e}")
                    logger.warning(f"Error inserting company summary for {ticker}: {e}")
            
            print(f"✅ Loaded {summary_count} company summaries")
            
            # Load stock prices
            print("📈 Loading stock prices...")
            price_count = 0
            for ticker, prices in self.stock_prices.items():
                try:
                    for date, row in prices.iterrows():
                        price_data = {
                            'ticker': ticker,
                            'date': date,
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume']
                        }
                        self.db_manager.insert_stock_price(price_data)
                        price_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Error inserting stock prices for {ticker}: {e}")
                    logger.warning(f"Error inserting stock prices for {ticker}: {e}")
            
            print(f"✅ Loaded {price_count} stock price records")
            print("✅ Data successfully loaded into database")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Failed to load data to database: {e}")
            print(f"   Stack trace: {traceback.format_exc()}")
            logger.error(f"Error loading data to database: {e}")
            print("⚠️  Warning: Database loading failed, but the application will continue with CSV exports")
            return False
        finally:
            if self.db_manager.connection:
                try:
                    self.db_manager.disconnect()
                    print("🔌 Database connection closed")
                except Exception as e:
                    print(f"⚠️  Warning: Error closing database connection: {e}")
    
    def run_unified_data_pipeline(self):
        """Run the complete unified data pipeline"""
        print("\n🚀 Starting unified data pipeline...")
        
        try:
            # Step 1: Collect all data once
            print("Step 1: Collecting all data...")
            if not self.collect_all_data():
                print("❌ ERROR: Failed to collect data. Exiting.")
                logger.error("Failed to collect data. Exiting.")
                return False
            
            # Step 2: Export PowerBI data
            print("Step 2: Exporting data to PowerBI CSV files...")
            powerbi_success = self.export_powerbi_data()
            if not powerbi_success:
                print("⚠️  Warning: PowerBI export failed, but continuing with database loading")
                logger.warning("PowerBI export failed, but continuing with database loading")
            
            # Step 3: Load data to database
            print("Step 3: Loading data to database...")
            db_success = self.load_data_to_database()
            if not db_success:
                print("⚠️  Warning: Database loading failed, but CSV exports were successful")
                logger.warning("Database loading failed, but CSV exports were successful")
            
            print("✅ Unified data pipeline completed!")
            return powerbi_success or db_success  # Return True if at least one succeeded
            
        except Exception as e:
            print(f"❌ ERROR: Failed during unified data pipeline: {e}")
            print(f"   Stack trace: {traceback.format_exc()}")
            logger.error(f"Error in unified data pipeline: {e}")
            return False
    

    
    def generate_data_summary(self):
        """Generate summary of loaded data"""
        try:
            print("\n" + "="*80)
            print("UNIFIED DATA PIPELINE SUMMARY")
            print("="*80)
            
            if self.company_summaries:
                print(f"\n📊 Collected data for {len(self.company_summaries)} companies:")
                for ticker, summary in self.company_summaries.items():
                    try:
                        print(f"  • {summary['company_name']} ({ticker})")
                        if summary.get('current_price'):
                            print(f"    Current Price: ${summary['current_price']:.2f}")
                        if summary.get('pe_ratio'):
                            print(f"    P/E Ratio: {summary['pe_ratio']:.2f}")
                        if summary.get('roe'):
                            print(f"    ROE: {summary['roe']:.2%}")
                        if summary.get('volatility'):
                            print(f"    Volatility: {summary['volatility']:.2%}")
                        print()
                    except Exception as e:
                        print(f"  • {ticker} - Error displaying summary: {e}")
            else:
                print("⚠️  No company data collected")
            
            print("📁 Output files created:")
            print("  • powerbi/data/ - CSV files for PowerBI dashboard")
            print("  • Database - Data loaded for main analysis")
            
            # Check if files actually exist
            powerbi_dir = 'powerbi/data'
            if os.path.exists(powerbi_dir):
                csv_files = [f for f in os.listdir(powerbi_dir) if f.endswith('.csv')]
                if csv_files:
                    print(f"  ✅ Found {len(csv_files)} CSV files in powerbi/data/")
                else:
                    print("  ⚠️  No CSV files found in powerbi/data/")
            else:
                print("  ❌ powerbi/data/ directory not found")
                
        except Exception as e:
            print(f"❌ ERROR: Failed to generate data summary: {e}")
            logger.error(f"Error generating data summary: {e}")

def main():
    """Main function to run the enhanced data fetching"""
    print("="*80)
    print("Enhanced Renewable Energy Data Fetcher")
    print("Unified Data Collection and Export System")
    print("="*80)
    
    try:
        # Validate configuration first
        if not validate_configuration():
            print("❌ Configuration validation failed. Exiting.")
            sys.exit(1)
        
        print("🚀 Starting data setup and collection...")
        logger.info("Starting data setup and collection...")
        
        # Initialize the data fetcher
        try:
            fetcher = EnhancedDataFetcher()
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize data fetcher: {e}")
            print(f"   Stack trace: {traceback.format_exc()}")
            logger.error(f"Failed to initialize data fetcher: {e}")
            sys.exit(1)
        
        # Run unified data pipeline
        success = fetcher.run_unified_data_pipeline()
        
        if success:
            # Generate summary
            try:
                fetcher.generate_data_summary()
            except Exception as e:
                print(f"⚠️  Warning: Could not generate data summary: {e}")
                logger.warning(f"Could not generate data summary: {e}")
            
            logger.info("Data setup completed successfully!")
            print("\n🎉 Data setup completed successfully!")
            print("📁 Check the 'powerbi/data/' directory for exported CSV files")
            print("📊 Check the database for stored data")
        else:
            logger.error("Data setup failed. Please check the logs for details.")
            print("\n❌ Data setup failed. Please check the logs for details.")
            print("📋 Check the console output above for specific error messages")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"   Stack trace: {traceback.format_exc()}")
        logger.critical(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 