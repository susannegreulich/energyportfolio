"""
Enhanced Data Fetching Module for Renewable Energy Investment Analysis
Unified data collection with export to both CSV (PowerBI) and database (main analysis)
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
from investment_analysis import InvestmentAnalyzer

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class EnhancedDataFetcher:
    """Enhanced data fetcher with unified collection and export"""
    
    def __init__(self):
        """Initialize the data fetcher"""
        self.db_manager = DatabaseManager()
        self.analyzer = InvestmentAnalyzer()
        
        # Create output directories
        os.makedirs('powerbi/data', exist_ok=True)
        
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
        
        # Store collected data for reuse
        self.collected_data = {}
        self.company_summaries = {}
        self.stock_prices = {}
        self.risk_metrics = {}
        
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
    
    def collect_all_data(self):
        """Collect all data once for both PowerBI export and database storage"""
        logger.info("Starting unified data collection for PowerBI export and database storage...")
        
        try:
            # Step 1: Collect stock data and company information
            logger.info("Step 1: Collecting stock data and company information...")
            for name, ticker in self.companies.items():
                logger.info(f"Collecting data for {name} ({ticker})...")
                
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
                            risk_metrics = self.analyzer.risk_analysis(returns)
                            self.risk_metrics[ticker] = risk_metrics
                            
                            # Update company summary with risk metrics
                            self.company_summaries[ticker]['volatility'] = risk_metrics['Volatility']
                            self.company_summaries[ticker]['sharpe_ratio'] = risk_metrics['Sharpe_Ratio']
                        
                        logger.info(f"Successfully collected data for {ticker}")
                        
                    else:
                        logger.warning(f"No valid data found for {ticker}")
                        
                except Exception as e:
                    logger.error(f"Error collecting data for {ticker}: {e}")
                    continue
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
            
            logger.info(f"Data collection completed. Collected data for {len(self.stock_prices)} companies.")
            return True
            
        except Exception as e:
            logger.error(f"Error in unified data collection: {e}")
            return False
    
    def export_powerbi_data(self):
        """Export all data to CSV files for PowerBI"""
        logger.info("Exporting data to PowerBI CSV files...")
        
        success = True
        
        # 1. Export company summary
        if self.company_summaries:
            try:
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
                logger.info(f"Company summary exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting company summary: {e}")
                success = False
        
        # 2. Export stock prices
        if self.stock_prices:
            try:
                all_prices = []
                
                for name, ticker in self.companies.items():
                    if ticker in self.stock_prices:
                        prices = self.stock_prices[ticker].copy()
                        prices['Ticker'] = ticker
                        prices['Company_Name'] = name
                        prices.reset_index(inplace=True)
                        all_prices.append(prices)
                
                if all_prices:
                    combined_prices = pd.concat(all_prices, ignore_index=True)
                    
                    # Add calculated fields
                    combined_prices['Year'] = combined_prices['Date'].dt.year
                    combined_prices['Month'] = combined_prices['Date'].dt.month
                    combined_prices['Quarter'] = combined_prices['Date'].dt.quarter
                    combined_prices['Day_of_Week'] = combined_prices['Date'].dt.dayofweek
                    
                    # Calculate returns
                    combined_prices['Daily_Return'] = combined_prices.groupby('Ticker')['Close'].pct_change()
                    combined_prices['Cumulative_Return'] = combined_prices.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod())
                    
                    output_path = 'powerbi/data/stock_prices.csv'
                    combined_prices.to_csv(output_path, index=False)
                    logger.info(f"Stock prices exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting stock prices: {e}")
                success = False
        
        # 3. Export risk metrics
        if self.risk_metrics:
            try:
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
                    logger.info(f"Risk metrics exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting risk metrics: {e}")
                success = False
        
        # 4. Export portfolio analysis
        if len(self.stock_prices) >= 2:
            try:
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
                    
                    if portfolio_data:
                        portfolio_df = pd.DataFrame(portfolio_data)
                        output_path = 'powerbi/data/portfolio_analysis.csv'
                        portfolio_df.to_csv(output_path, index=False)
                        logger.info(f"Portfolio analysis exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting portfolio analysis: {e}")
                success = False
        
        # 5. Export technical indicators
        if self.stock_prices:
            try:
                all_technical_data = []
                
                for name, ticker in self.companies.items():
                    if ticker in self.stock_prices:
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
                
                if all_technical_data:
                    combined_technical = pd.concat(all_technical_data, ignore_index=True)
                    output_path = 'powerbi/data/technical_indicators.csv'
                    combined_technical.to_csv(output_path, index=False)
                    logger.info(f"Technical indicators exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting technical indicators: {e}")
                success = False
        
        # 6. Export correlation matrix
        if len(self.stock_prices) >= 2:
            try:
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
                        logger.info(f"Correlation matrix exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting correlation matrix: {e}")
                success = False
        
        if success:
            logger.info("All PowerBI data exported successfully!")
            print("\nPowerBI Data Export Summary:")
            print("="*40)
            print("Files exported to 'powerbi/data/' directory:")
            print("• company_summary.csv")
            print("• stock_prices.csv")
            print("• risk_metrics.csv")
            print("• portfolio_analysis.csv")
            print("• technical_indicators.csv")
            print("• correlation_matrix.csv")
        
        return success
    
    def load_data_to_database(self):
        """Load collected data into database for main analysis"""
        logger.info("Loading collected data into database...")
        
        try:
            # Create database and tables
            if not create_database():
                logger.warning("Failed to create database, continuing without database")
                return True
            
            # Connect to database
            if not self.db_manager.connect():
                logger.warning("Failed to connect to database, continuing without database")
                return True
            
            # Load company summaries
            for ticker, summary in self.company_summaries.items():
                try:
                    self.db_manager.insert_company_summary(summary)
                except Exception as e:
                    logger.warning(f"Error inserting company summary for {ticker}: {e}")
            
            # Load stock prices
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
                except Exception as e:
                    logger.warning(f"Error inserting stock prices for {ticker}: {e}")
            
            logger.info("Data successfully loaded into database")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data to database: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def run_unified_data_pipeline(self):
        """Run the complete unified data pipeline"""
        logger.info("Starting unified data pipeline...")
        
        # Step 1: Collect all data once
        logger.info("Step 1: Collecting all data...")
        if not self.collect_all_data():
            logger.error("Failed to collect data. Exiting.")
            return False
        
        # Step 2: Export PowerBI data
        logger.info("Step 2: Exporting data to PowerBI CSV files...")
        if not self.export_powerbi_data():
            logger.warning("PowerBI export failed, but continuing with database loading")
        
        # Step 3: Load data to database
        logger.info("Step 3: Loading data to database...")
        self.load_data_to_database()
        
        logger.info("Unified data pipeline completed successfully!")
        return True
    
    def generate_data_summary(self):
        """Generate summary of loaded data"""
        print("\n" + "="*80)
        print("UNIFIED DATA PIPELINE SUMMARY")
        print("="*80)
        
        if self.company_summaries:
            print(f"\nCollected data for {len(self.company_summaries)} companies:")
            for ticker, summary in self.company_summaries.items():
                print(f"  • {summary['company_name']} ({ticker})")
                if summary['current_price']:
                    print(f"    Current Price: ${summary['current_price']:.2f}")
                if summary['pe_ratio']:
                    print(f"    P/E Ratio: {summary['pe_ratio']:.2f}")
                if summary['roe']:
                    print(f"    ROE: {summary['roe']:.2%}")
                print()
        else:
            print("No data collected")
        
        print("Output files created:")
        print("• powerbi/data/ - CSV files for PowerBI dashboard")
        print("• Database - Data loaded for main analysis")

def main():
    """Main function to run the enhanced data fetching"""
    fetcher = EnhancedDataFetcher()
    
    print("Enhanced Renewable Energy Data Fetcher")
    print("Unified Data Collection and Export System")
    print("="*60)
    
    # Run unified data pipeline
    success = fetcher.run_unified_data_pipeline()
    
    if success:
        # Generate summary
        fetcher.generate_data_summary()
        print("\nUnified data pipeline completed successfully!")
    else:
        print("\nUnified data pipeline failed. Please check the logs for details.")

if __name__ == "__main__":
    main() 