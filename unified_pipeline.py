#!/usr/bin/env python3
"""
Unified Investment Analysis Pipeline
Zero duplication, PowerBI export at the end using all analysis results
"""

import sys
import traceback
import logging
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('unified_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print(f"\n{'='*60}")
    print("CHECKING PREREQUISITES")
    print(f"{'='*60}")
    
    # Check if config.py exists and has been configured
    config_exists = os.path.exists("config.py")
    
    if not config_exists:
        print("❌ config.py not found.")
        print("Please run setup.py first: python3 setup.py")
        return False
    
    # Check if config.py has been properly configured. it has the necessary API keys.
    try:
        with open("config.py", "r") as f:
            config_content = f.read()
            if "your_finnhub_api_key_here" in config_content:
                print("⚠️  config.py appears to have default values.")
                print("Please run setup.py first: python3 setup.py")
                return False
    except Exception as e:
        print(f"❌ Error reading config.py: {e}")
        return False
    
    # Check if required packages are installed
    try:
        import pandas
        import numpy
        import yfinance
        import scipy
        import matplotlib
        import seaborn
        import requests
        from textblob import TextBlob
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False
    
    # Check database connection (optional)
    try:
        import psycopg2
        from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.close()
        print("✅ Database connection verified")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("Pipeline will continue without database storage")
    
    print("✅ All prerequisites verified")
    return True

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from scipy import stats
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import requests
    from textblob import TextBlob
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please install required packages: pip install pandas numpy yfinance scipy matplotlib seaborn requests textblob")
    sys.exit(1)

try:
    from config import *
    from database import DatabaseManager
except ImportError as e:
    print(f"ERROR: Failed to import config or database modules: {e}")
    print("Please ensure config.py and database.py are in the same directory")
    sys.exit(1)

class UnifiedInvestmentPipeline:
    """Unified pipeline with zero duplication and PowerBI export at the end"""
    
    def __init__(self):
        """Initialize the unified pipeline"""
        try:
            self.db_manager = DatabaseManager()
            
            # Companies to analyze
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
            
            # Data storage
            self.stock_prices = {}
            self.company_summaries = {}
            self.risk_metrics = {}
            self.technical_indicators = {}
            self.fundamental_metrics = {}
            self.portfolio_analysis = {}
            self.correlation_matrix = None
            
            # Create necessary directories
            directories = ['powerbi/data', 'analysis', 'reports', 'charts', 'logs']
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory: {directory}")
            
            print("✅ UnifiedInvestmentPipeline initialized successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize UnifiedInvestmentPipeline: {e}")
            logger.error(f"Initialization error: {traceback.format_exc()}")
            raise
    
    def fetch_stock_data(self, ticker, start_date=None, end_date=None):
        """Fetch comprehensive stock data including financial statements"""
        try:
            print(f"📊 Fetching data for {ticker}...")
            
            # Use yfinance for comprehensive data
            stock = yf.Ticker(ticker)
            
            # Fetch historical prices - get maximum available data
            history = stock.history(period="max")
            if history.empty:
                print(f"❌ No historical data available for {ticker}")
                return None
            
            # Fetch financial statements
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cash_flow
            except Exception as e:
                print(f"⚠️  Warning: Could not fetch financial statements for {ticker}: {e}")
                income_stmt = None
                balance_sheet = None
                cash_flow = None
            
            # Get company info
            info = stock.info
            
            return {
                'history': history,
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'info': info
            }
            
        except Exception as e:
            print(f"❌ ERROR: Failed to fetch data for {ticker}: {e}")
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_financial_metrics(self, stock_data):
        """Calculate comprehensive financial metrics using financial statements when available"""
        if not stock_data or stock_data['income_stmt'] is None:
            return {}
        
        try:
            income_stmt = stock_data['income_stmt']
            balance_sheet = stock_data['balance_sheet']
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
            
            # Basic ratios from yfinance info (fallback)
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
    
    def calculate_returns(self, prices):
        """Calculate daily and cumulative returns"""
        try:
            if prices is None or prices.empty:
                return None, None
            
            if hasattr(prices, 'columns'):
                if 'Close' not in prices.columns:
                    return None, None
                price_series = prices['Close']
            else:
                price_series = prices
            
            returns = price_series.pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            return returns, cumulative_returns
        except Exception as e:
            print(f"ERROR: Failed to calculate returns: {e}")
            return None, None
    
    def calculate_volatility(self, returns, window=252):
        """Calculate rolling volatility"""
        try:
            if returns is None or returns.empty:
                return None
            return returns.rolling(window=window).std() * np.sqrt(252)
        except Exception as e:
            print(f"ERROR: Failed to calculate volatility: {e}")
            return None
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02, window=252):
        """Calculate Sharpe ratio"""
        try:
            if returns is None or returns.empty:
                return None
            excess_returns = returns - risk_free_rate/252
            sharpe = excess_returns.rolling(window=window).mean() / returns.rolling(window=window).std()
            return sharpe * np.sqrt(252)
        except Exception as e:
            print(f"ERROR: Failed to calculate Sharpe ratio: {e}")
            return None
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        try:
            if cumulative_returns is None or cumulative_returns.empty:
                return None
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return drawdown.min()
        except Exception as e:
            print(f"ERROR: Failed to calculate maximum drawdown: {e}")
            return None
    
    def calculate_var(self, returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        try:
            if returns is None or returns.empty:
                return None
            return np.percentile(returns, confidence_level * 100)
        except Exception as e:
            print(f"ERROR: Failed to calculate VaR: {e}")
            return None
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk"""
        try:
            if returns is None or returns.empty:
                return None
            var = self.calculate_var(returns, confidence_level)
            if var is None:
                return None
            return returns[returns <= var].mean()
        except Exception as e:
            print(f"ERROR: Failed to calculate CVaR: {e}")
            return None
    
    def technical_analysis(self, prices):
        """Perform comprehensive technical analysis"""
        try:
            if prices is None or prices.empty:
                return {}
            
            if hasattr(prices, 'columns'):
                if 'Close' not in prices.columns:
                    return {}
                price_series = prices['Close']
            else:
                price_series = prices
            
            # Moving averages
            sma_20 = price_series.rolling(window=20).mean()
            sma_50 = price_series.rolling(window=50).mean()
            sma_200 = price_series.rolling(window=200).mean()
            
            # RSI
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = price_series.ewm(span=12).mean()
            ema_26 = price_series.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_middle = price_series.rolling(window=20).mean()
            bb_std = price_series.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower
            }
        except Exception as e:
            print(f"ERROR: Failed to perform technical analysis: {e}")
            return {}
    
    def risk_analysis(self, returns, benchmark_returns=None):
        """Perform comprehensive risk analysis"""
        try:
            if returns is None or returns.empty:
                return {}
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)
            annual_return = returns.mean() * 252
            
            # Sharpe ratio
            if volatility > 0:
                sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            max_drawdown = self.calculate_max_drawdown(cumulative_returns)
            
            # Value at Risk
            var_95 = self.calculate_var(returns, 0.05)
            cvar_95 = self.calculate_cvar(returns, 0.05)
            
            # Beta calculation if benchmark provided
            beta = None
            if benchmark_returns is not None and not benchmark_returns.empty:
                try:
                    benchmark_var = benchmark_returns.var()
                    if benchmark_var > 0:
                        beta = returns.cov(benchmark_returns) / benchmark_var
                except Exception as e:
                    print(f"ERROR: Failed to calculate beta: {e}")
            
            return {
                'Volatility': volatility,
                'Annual_Return': annual_return,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'VaR_95': var_95,
                'CVaR_95': cvar_95,
                'Beta': beta
            }
        except Exception as e:
            print(f"ERROR: Failed to perform risk analysis: {e}")
            return {}
    
    def portfolio_optimization(self, returns_data, method='sharpe'):
        """Optimize portfolio weights using different methods"""
        try:
            if returns_data is None or returns_data.empty:
                return None
            
            n_assets = len(returns_data.columns)
            if n_assets == 0:
                return None
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            
            if cov_matrix.isnull().any().any():
                return None
            
            if method == 'sharpe':
                def objective(weights):
                    try:
                        portfolio_return = np.sum(expected_returns * weights)
                        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        if portfolio_vol > 0:
                            sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
                        else:
                            sharpe = 0
                        return -sharpe
                    except Exception as e:
                        return 1e6
                
            elif method == 'min_variance':
                def objective(weights):
                    try:
                        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return portfolio_vol
                    except Exception as e:
                        return 1e6
                
            elif method == 'max_return':
                def objective(weights):
                    try:
                        portfolio_return = np.sum(expected_returns * weights)
                        return -portfolio_return
                    except Exception as e:
                        return 1e6
            else:
                return None
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            try:
                result = minimize(objective, initial_weights, method='SLSQP', 
                                bounds=bounds, constraints=constraints)
                
                if result.success:
                    optimal_weights = result.x
                    portfolio_return = np.sum(expected_returns * optimal_weights)
                    portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                    
                    if portfolio_vol > 0:
                        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
                    else:
                        sharpe_ratio = 0
                    
                    return {
                        'weights': optimal_weights,
                        'expected_return': portfolio_return,
                        'volatility': portfolio_vol,
                        'sharpe_ratio': sharpe_ratio
                    }
                else:
                    return None
            except Exception as e:
                return None
                
        except Exception as e:
            return None
    
    def collect_all_data(self):
        """Step 1: Collect all raw data"""
        print("\n📈 Step 1: Collecting all raw data...")
        
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
                    self.fundamental_metrics[ticker] = financial_metrics
                    
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
                        **financial_metrics  # Add all calculated financial metrics
                    }
                    
                    successful_companies += 1
                    print(f"✅ Successfully collected data for {ticker}")
                else:
                    print(f"❌ Failed to collect data for {ticker}")
                    
            except Exception as e:
                print(f"❌ ERROR: Failed to process {ticker}: {e}")
                logger.error(f"Error processing {ticker}: {e}")
        
        print(f"\n✅ Data collection completed. Successfully processed {successful_companies}/{len(self.companies)} companies.")
        return successful_companies > 0
    
    def perform_technical_analysis(self):
        """Step 2: Perform technical analysis for all companies"""
        print("\n📊 Step 2: Performing technical analysis...")
        
        all_technical_data = []
        
        for ticker, prices in self.stock_prices.items():
            try:
                technical = self.technical_analysis(prices)
                
                if technical:
                    # Create DataFrame with technical indicators
                    tech_df = pd.DataFrame({
                        'Date': prices.index,
                        'Ticker': ticker,
                        'Close': prices['Close'],
                        'Volume': prices['Volume'],
                        'SMA_20': technical['sma_20'],
                        'SMA_50': technical['sma_50'],
                        'SMA_200': technical['sma_200'],
                        'RSI': technical['rsi'],
                        'MACD': technical['macd'],
                        'MACD_Signal': technical['macd_signal'],
                        'MACD_Histogram': technical['macd_histogram'],
                        'BB_Upper': technical['bb_upper'],
                        'BB_Middle': technical['bb_middle'],
                        'BB_Lower': technical['bb_lower']
                    })
                    
                    all_technical_data.append(tech_df)
                    print(f"✅ Technical analysis completed for {ticker}")
                
            except Exception as e:
                print(f"❌ ERROR: Technical analysis failed for {ticker}: {e}")
                logger.error(f"Technical analysis error for {ticker}: {e}")
        
        if all_technical_data:
            self.technical_indicators = pd.concat(all_technical_data, ignore_index=True)
            print(f"✅ Technical analysis completed for {len(self.stock_prices)} companies")
        else:
            print("❌ No technical analysis data generated")
    
    def perform_risk_analysis(self):
        """Step 3: Perform risk analysis for all companies"""
        print("\n⚠️  Step 3: Performing risk analysis...")
        
        all_risk_data = []
        
        for ticker, prices in self.stock_prices.items():
            try:
                returns = prices['Close'].pct_change().dropna()
                
                if len(returns) > 30:  # Need sufficient data
                    risk_metrics = self.risk_analysis(returns)
                    
                    if risk_metrics:
                        risk_metrics['Ticker'] = ticker
                        risk_metrics['Company_Name'] = self.companies.get(ticker, ticker)
                        risk_metrics['Analysis_Date'] = datetime.now().date()
                        
                        # Add risk categories
                        volatility = risk_metrics.get('Volatility', 0)
                        if volatility <= 0.2:
                            risk_metrics['Risk_Level'] = 'Low'
                        elif volatility <= 0.4:
                            risk_metrics['Risk_Level'] = 'Medium'
                        elif volatility <= 0.6:
                            risk_metrics['Risk_Level'] = 'High'
                        else:
                            risk_metrics['Risk_Level'] = 'Very High'
                        
                        sharpe = risk_metrics.get('Sharpe_Ratio', 0)
                        if sharpe <= 0:
                            risk_metrics['Sharpe_Category'] = 'Poor'
                        elif sharpe <= 0.5:
                            risk_metrics['Sharpe_Category'] = 'Fair'
                        elif sharpe <= 1:
                            risk_metrics['Sharpe_Category'] = 'Good'
                        else:
                            risk_metrics['Sharpe_Category'] = 'Excellent'
                        
                        all_risk_data.append(risk_metrics)
                        self.risk_metrics[ticker] = risk_metrics
                        
                        # Update company summary with risk metrics
                        self.company_summaries[ticker]['volatility'] = risk_metrics['Volatility']
                        self.company_summaries[ticker]['sharpe_ratio'] = risk_metrics['Sharpe_Ratio']
                        self.company_summaries[ticker]['beta'] = risk_metrics.get('Beta')
                        
                        print(f"✅ Risk analysis completed for {ticker}")
                
            except Exception as e:
                print(f"❌ ERROR: Risk analysis failed for {ticker}: {e}")
                logger.error(f"Risk analysis error for {ticker}: {e}")
        
        if all_risk_data:
            print(f"✅ Risk analysis completed for {len(all_risk_data)} companies")
        else:
            print("❌ No risk analysis data generated")
    
    def perform_portfolio_analysis(self):
        """Step 4: Perform portfolio optimization analysis"""
        print("\n📊 Step 4: Performing portfolio optimization analysis...")
        
        if len(self.stock_prices) < 2:
            print("⚠️  Need at least 2 companies for portfolio analysis")
            return
        
        try:
            # Get returns data for all companies
            all_returns = {}
            for ticker, prices in self.stock_prices.items():
                returns = prices['Close'].pct_change().dropna()
                all_returns[ticker] = returns
            
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                print("⚠️  No valid returns data for portfolio analysis")
                return
            
            print(f"📊 Portfolio analysis with {len(returns_df)} rows and {len(returns_df.columns)} assets")
            
            # Perform portfolio optimization
            optimization_results = {}
            
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
                        result = self.portfolio_optimization(returns_df, method)
                    
                    if result:
                        optimization_results[method] = result
                        print(f"✅ {method} optimization completed")
                    
                except Exception as e:
                    print(f"❌ ERROR: {method} optimization failed: {e}")
            
            self.portfolio_analysis = optimization_results
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            self.correlation_matrix = correlation_matrix
            
            print(f"✅ Portfolio analysis completed with {len(optimization_results)} methods")
            
        except Exception as e:
            print(f"❌ ERROR: Portfolio analysis failed: {e}")
            logger.error(f"Portfolio analysis error: {e}")
    
    def export_powerbi_data(self):
        """Step 5: Export all analysis results to PowerBI CSV files"""
        print("\n📊 Step 5: Exporting all analysis results to PowerBI...")
        
        exported_files = []
        
        try:
            # 1. Export company summary with all metrics
            if self.company_summaries:
                print("📋 Exporting company summary...")
                summary_df = pd.DataFrame(list(self.company_summaries.values()))
                
                # Add calculated fields
                summary_df['Market_Cap_Billions'] = summary_df['market_cap'] / 1e9 if 'market_cap' in summary_df.columns else None
                
                # Handle PE ratio categorization (check both possible column names)
                pe_column = 'pe_ratio' if 'pe_ratio' in summary_df.columns else 'PE_Ratio'
                if pe_column in summary_df.columns:
                    summary_df['PE_Category'] = pd.cut(summary_df[pe_column], 
                                                      bins=[0, 15, 25, 50, float('inf')], 
                                                      labels=['Value', 'Fair Value', 'Growth', 'High Growth'])
                
                # Handle volatility categorization
                if 'volatility' in summary_df.columns:
                    summary_df['Risk_Category'] = pd.cut(summary_df['volatility'], 
                                                        bins=[0, 0.2, 0.4, 0.6, float('inf')], 
                                                        labels=['Low', 'Medium', 'High', 'Very High'])
                
                output_path = 'powerbi/data/company_summary.csv'
                summary_df.to_csv(output_path, index=False)
                exported_files.append('company_summary.csv')
                print(f"✅ Company summary exported to {output_path}")
            
            # 2. Export stock prices
            if self.stock_prices:
                print("📈 Exporting stock prices...")
                all_prices = []
                
                for name, ticker in self.companies.items():
                    if ticker in self.stock_prices:
                        prices = self.stock_prices[ticker].copy()
                        prices['Ticker'] = ticker
                        prices['Company_Name'] = name
                        prices.reset_index(inplace=True)
                        
                        if 'Date' not in prices.columns:
                            date_col = prices.columns[0]
                            prices.rename(columns={date_col: 'Date'}, inplace=True)
                        
                        prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce')
                        
                        # Add calculated fields
                        prices['Year'] = prices['Date'].dt.year
                        prices['Month'] = prices['Date'].dt.month
                        prices['Quarter'] = prices['Date'].dt.quarter
                        prices['Day_of_Week'] = prices['Date'].dt.dayofweek
                        prices['Daily_Return'] = prices.groupby('Ticker')['Close'].pct_change()
                        prices['Cumulative_Return'] = prices.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod())
                        
                        all_prices.append(prices)
                
                if all_prices:
                    combined_prices = pd.concat(all_prices, ignore_index=True)
                    output_path = 'powerbi/data/stock_prices.csv'
                    combined_prices.to_csv(output_path, index=False)
                    exported_files.append('stock_prices.csv')
                    print(f"✅ Stock prices exported to {output_path}")
            
            # 3. Export risk metrics
            if self.risk_metrics:
                print("⚠️  Exporting risk metrics...")
                risk_df = pd.DataFrame(list(self.risk_metrics.values()))
                output_path = 'powerbi/data/risk_metrics.csv'
                risk_df.to_csv(output_path, index=False)
                exported_files.append('risk_metrics.csv')
                print(f"✅ Risk metrics exported to {output_path}")
            
            # 4. Export technical indicators
            if not self.technical_indicators.empty:
                print("📊 Exporting technical indicators...")
                output_path = 'powerbi/data/technical_indicators.csv'
                self.technical_indicators.to_csv(output_path, index=False)
                exported_files.append('technical_indicators.csv')
                print(f"✅ Technical indicators exported to {output_path}")
            
            # 5. Export portfolio analysis
            if self.portfolio_analysis:
                print("📊 Exporting portfolio analysis...")
                portfolio_data = []
                
                for method, result in self.portfolio_analysis.items():
                    for i, ticker in enumerate(self.stock_prices.keys()):
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
                    exported_files.append('portfolio_analysis.csv')
                    print(f"✅ Portfolio analysis exported to {output_path}")
            else:
                print("⚠️  No portfolio analysis data to export")
            
            # 6. Export correlation matrix
            if self.correlation_matrix is not None:
                print("📊 Exporting correlation matrix...")
                correlation_data = []
                
                for i, ticker1 in enumerate(self.correlation_matrix.index):
                    for j, ticker2 in enumerate(self.correlation_matrix.columns):
                        correlation_data.append({
                            'Ticker1': ticker1,
                            'Ticker2': ticker2,
                            'Correlation': self.correlation_matrix.iloc[i, j],
                            'Correlation_Abs': abs(self.correlation_matrix.iloc[i, j]),
                            'Correlation_Category': 'High' if abs(self.correlation_matrix.iloc[i, j]) > 0.7 else 
                                                  'Medium' if abs(self.correlation_matrix.iloc[i, j]) > 0.3 else 'Low'
                        })
                
                if correlation_data:
                    correlation_df = pd.DataFrame(correlation_data)
                    output_path = 'powerbi/data/correlation_matrix.csv'
                    correlation_df.to_csv(output_path, index=False)
                    exported_files.append('correlation_matrix.csv')
                    print(f"✅ Correlation matrix exported to {output_path}")
            else:
                print("⚠️  No correlation matrix data to export")
            
            print(f"\n✅ PowerBI export completed successfully!")
            print(f"📁 Exported {len(exported_files)} files to 'powerbi/data/' directory:")
            for file in exported_files:
                print(f"   • {file}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Failed to export PowerBI data: {e}")
            logger.error(f"PowerBI export error: {e}")
            return False
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report with all analysis results"""
        print("\n📝 Step 6: Generating comprehensive markdown report...")
        
        try:
            report_content = []
            
            # Header
            report_content.append("# Renewable Energy Investment Analysis Report")
            report_content.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"**Analysis Period:** Maximum available historical data")
            report_content.append(f"**Companies Analyzed:** {len(self.company_summaries)}")
            report_content.append("\n---\n")
            
            # Executive Summary
            report_content.append("## 📊 Executive Summary")
            report_content.append("\nThis report provides a comprehensive analysis of renewable energy companies for investment purposes. The analysis includes:")
            report_content.append("- **Fundamental Analysis:** Financial ratios and metrics")
            report_content.append("- **Technical Analysis:** Price patterns and indicators")
            report_content.append("- **Risk Analysis:** Volatility, Sharpe ratios, and risk metrics")
            report_content.append("- **Portfolio Optimization:** Multiple optimization strategies")
            report_content.append("- **Correlation Analysis:** Inter-company relationships")
            
            # Company Overview
            if self.company_summaries:
                report_content.append("\n## 🏢 Company Overview")
                report_content.append("\n| Company | Ticker | Current Price | Market Cap | P/E Ratio | ROE | Volatility |")
                report_content.append("|---------|--------|---------------|------------|-----------|-----|------------|")
                
                for ticker, summary in self.company_summaries.items():
                    try:
                        company_name = summary.get('company_name', ticker)
                        current_price = summary.get('current_price', 'N/A')
                        market_cap = summary.get('market_cap', 'N/A')
                        pe_ratio = summary.get('pe_ratio', summary.get('PE_Ratio', 'N/A'))
                        roe = summary.get('roe', summary.get('ROE', 'N/A'))
                        volatility = summary.get('volatility', 'N/A')
                        
                        # Format values
                        if isinstance(current_price, (int, float)):
                            current_price = f"${current_price:.2f}"
                        if isinstance(market_cap, (int, float)):
                            market_cap = f"${market_cap/1e9:.1f}B"
                        if isinstance(pe_ratio, (int, float)):
                            pe_ratio = f"{pe_ratio:.2f}"
                        if isinstance(roe, (int, float)):
                            roe = f"{roe:.2%}"
                        if isinstance(volatility, (int, float)):
                            volatility = f"{volatility:.2%}"
                        
                        report_content.append(f"| {company_name} | {ticker} | {current_price} | {market_cap} | {pe_ratio} | {roe} | {volatility} |")
                    except Exception as e:
                        report_content.append(f"| {ticker} | {ticker} | Error | Error | Error | Error | Error |")
            
            # Fundamental Analysis
            if self.fundamental_metrics:
                report_content.append("\n## 📈 Fundamental Analysis")
                report_content.append("\n### Key Financial Metrics")
                report_content.append("\n| Company | P/E Ratio | P/B Ratio | Debt/Equity | Current Ratio | ROE | ROA | Profit Margin |")
                report_content.append("|---------|-----------|-----------|-------------|---------------|-----|-----|---------------|")
                
                for ticker, metrics in self.fundamental_metrics.items():
                    try:
                        company_name = self.companies.get(ticker, ticker)
                        pe_ratio = metrics.get('PE_Ratio', 'N/A')
                        pb_ratio = metrics.get('PB_Ratio', 'N/A')
                        debt_equity = metrics.get('Debt_to_Equity', 'N/A')
                        current_ratio = metrics.get('Current_Ratio', 'N/A')
                        roe = metrics.get('ROE', 'N/A')
                        roa = metrics.get('ROA', 'N/A')
                        profit_margin = metrics.get('Profit_Margin', 'N/A')
                        
                        # Format values
                        if isinstance(pe_ratio, (int, float)):
                            pe_ratio = f"{pe_ratio:.2f}"
                        if isinstance(pb_ratio, (int, float)):
                            pb_ratio = f"{pb_ratio:.2f}"
                        if isinstance(debt_equity, (int, float)):
                            debt_equity = f"{debt_equity:.2f}"
                        if isinstance(current_ratio, (int, float)):
                            current_ratio = f"{current_ratio:.2f}"
                        if isinstance(roe, (int, float)):
                            roe = f"{roe:.2%}"
                        if isinstance(roa, (int, float)):
                            roa = f"{roa:.2%}"
                        if isinstance(profit_margin, (int, float)):
                            profit_margin = f"{profit_margin:.2%}"
                        
                        report_content.append(f"| {company_name} | {pe_ratio} | {pb_ratio} | {debt_equity} | {current_ratio} | {roe} | {roa} | {profit_margin} |")
                    except Exception as e:
                        report_content.append(f"| {ticker} | Error | Error | Error | Error | Error | Error | Error |")
            
            # Risk Analysis
            if self.risk_metrics:
                report_content.append("\n## ⚠️ Risk Analysis")
                report_content.append("\n### Risk Metrics Summary")
                report_content.append("\n| Company | Volatility | Annual Return | Sharpe Ratio | Max Drawdown | VaR (95%) | Beta | Risk Level |")
                report_content.append("|---------|------------|---------------|--------------|-------------|------------|------|------------|")
                
                for ticker, metrics in self.risk_metrics.items():
                    try:
                        company_name = self.companies.get(ticker, ticker)
                        volatility = metrics.get('Volatility', 'N/A')
                        annual_return = metrics.get('Annual_Return', 'N/A')
                        sharpe = metrics.get('Sharpe_Ratio', 'N/A')
                        max_drawdown = metrics.get('Max_Drawdown', 'N/A')
                        var_95 = metrics.get('VaR_95', 'N/A')
                        beta = metrics.get('Beta', 'N/A')
                        risk_level = metrics.get('Risk_Level', 'N/A')
                        
                        # Format values
                        if isinstance(volatility, (int, float)):
                            volatility = f"{volatility:.2%}"
                        if isinstance(annual_return, (int, float)):
                            annual_return = f"{annual_return:.2%}"
                        if isinstance(sharpe, (int, float)):
                            sharpe = f"{sharpe:.2f}"
                        if isinstance(max_drawdown, (int, float)):
                            max_drawdown = f"{max_drawdown:.2%}"
                        if isinstance(var_95, (int, float)):
                            var_95 = f"{var_95:.2%}"
                        if isinstance(beta, (int, float)):
                            beta = f"{beta:.2f}"
                        
                        report_content.append(f"| {company_name} | {volatility} | {annual_return} | {sharpe} | {max_drawdown} | {var_95} | {beta} | {risk_level} |")
                    except Exception as e:
                        report_content.append(f"| {ticker} | Error | Error | Error | Error | Error | Error | Error |")
                
                # Risk Analysis Insights
                report_content.append("\n### Risk Analysis Insights")
                
                # Find best and worst performers
                if self.risk_metrics:
                    sharpe_ratios = {ticker: metrics.get('Sharpe_Ratio', 0) for ticker, metrics in self.risk_metrics.items()}
                    volatilities = {ticker: metrics.get('Volatility', 0) for ticker, metrics in self.risk_metrics.items()}
                    
                    if sharpe_ratios:
                        best_sharpe = max(sharpe_ratios.items(), key=lambda x: x[1] if x[1] is not None else -999)
                        worst_sharpe = min(sharpe_ratios.items(), key=lambda x: x[1] if x[1] is not None else 999)
                        
                        report_content.append(f"\n- **Best Risk-Adjusted Return:** {self.companies.get(best_sharpe[0], best_sharpe[0])} (Sharpe: {best_sharpe[1]:.2f})")
                        report_content.append(f"- **Worst Risk-Adjusted Return:** {self.companies.get(worst_sharpe[0], worst_sharpe[0])} (Sharpe: {worst_sharpe[1]:.2f})")
                    
                    if volatilities:
                        lowest_vol = min(volatilities.items(), key=lambda x: x[1] if x[1] is not None else 999)
                        highest_vol = max(volatilities.items(), key=lambda x: x[1] if x[1] is not None else -999)
                        
                        report_content.append(f"- **Lowest Volatility:** {self.companies.get(lowest_vol[0], lowest_vol[0])} ({lowest_vol[1]:.2%})")
                        report_content.append(f"- **Highest Volatility:** {self.companies.get(highest_vol[0], highest_vol[0])} ({highest_vol[1]:.2%})")
            
            # Portfolio Analysis
            if self.portfolio_analysis:
                report_content.append("\n## 📊 Portfolio Analysis")
                report_content.append("\n### Portfolio Optimization Results")
                report_content.append("\n| Optimization Method | Expected Return | Volatility | Sharpe Ratio |")
                report_content.append("|-------------------|-----------------|------------|--------------|")
                
                for method, result in self.portfolio_analysis.items():
                    try:
                        method_name = method.replace('_', ' ').title()
                        expected_return = result.get('expected_return', 0)
                        volatility = result.get('volatility', 0)
                        sharpe = result.get('sharpe_ratio', 0)
                        
                        report_content.append(f"| {method_name} | {expected_return:.2%} | {volatility:.2%} | {sharpe:.2f} |")
                    except Exception as e:
                        report_content.append(f"| {method} | Error | Error | Error |")
                
                # Portfolio Weights
                report_content.append("\n### Optimal Portfolio Weights")
                report_content.append("\n| Company | Sharpe Weight | Min Variance Weight | Max Return Weight | Equal Weight |")
                report_content.append("|---------|---------------|-------------------|------------------|--------------|")
                
                if self.stock_prices:
                    tickers = list(self.stock_prices.keys())
                    
                    for i, ticker in enumerate(tickers):
                        try:
                            company_name = self.companies.get(ticker, ticker)
                            
                            # Get weights from different methods
                            sharpe_weight = self.portfolio_analysis.get('sharpe', {}).get('weights', [0] * len(tickers))[i] if i < len(self.portfolio_analysis.get('sharpe', {}).get('weights', [])) else 0
                            min_var_weight = self.portfolio_analysis.get('min_variance', {}).get('weights', [0] * len(tickers))[i] if i < len(self.portfolio_analysis.get('min_variance', {}).get('weights', [])) else 0
                            max_return_weight = self.portfolio_analysis.get('max_return', {}).get('weights', [0] * len(tickers))[i] if i < len(self.portfolio_analysis.get('max_return', {}).get('weights', [])) else 0
                            equal_weight = 1.0 / len(tickers)
                            
                            report_content.append(f"| {company_name} | {sharpe_weight:.2%} | {min_var_weight:.2%} | {max_return_weight:.2%} | {equal_weight:.2%} |")
                        except Exception as e:
                            report_content.append(f"| {ticker} | Error | Error | Error | Error |")
            
            # Correlation Analysis
            if self.correlation_matrix is not None:
                report_content.append("\n## 🔗 Correlation Analysis")
                report_content.append("\n### Correlation Matrix")
                report_content.append("\n| Company 1 | Company 2 | Correlation | Category |")
                report_content.append("|-----------|-----------|-------------|----------|")
                
                for i, ticker1 in enumerate(self.correlation_matrix.index):
                    for j, ticker2 in enumerate(self.correlation_matrix.columns):
                        if i < j:  # Only show upper triangle
                            try:
                                company1 = self.companies.get(ticker1, ticker1)
                                company2 = self.companies.get(ticker2, ticker2)
                                correlation = self.correlation_matrix.iloc[i, j]
                                
                                if abs(correlation) > 0.7:
                                    category = "High"
                                elif abs(correlation) > 0.3:
                                    category = "Medium"
                                else:
                                    category = "Low"
                                
                                report_content.append(f"| {company1} | {company2} | {correlation:.3f} | {category} |")
                            except Exception as e:
                                report_content.append(f"| {ticker1} | {ticker2} | Error | Error |")
                
                # Correlation Insights
                report_content.append("\n### Correlation Insights")
                
                # Find highest and lowest correlations
                correlations = []
                for i, ticker1 in enumerate(self.correlation_matrix.index):
                    for j, ticker2 in enumerate(self.correlation_matrix.columns):
                        if i < j:
                            correlations.append((ticker1, ticker2, self.correlation_matrix.iloc[i, j]))
                
                if correlations:
                    highest_corr = max(correlations, key=lambda x: abs(x[2]))
                    lowest_corr = min(correlations, key=lambda x: abs(x[2]))
                    
                    report_content.append(f"\n- **Highest Correlation:** {self.companies.get(highest_corr[0], highest_corr[0])} & {self.companies.get(highest_corr[1], highest_corr[1])} ({highest_corr[2]:.3f})")
                    report_content.append(f"- **Lowest Correlation:** {self.companies.get(lowest_corr[0], lowest_corr[0])} & {self.companies.get(lowest_corr[1], lowest_corr[1])} ({lowest_corr[2]:.3f})")
            
            # Technical Analysis Summary
            if not self.technical_indicators.empty:
                report_content.append("\n## 📈 Technical Analysis")
                report_content.append("\n### Technical Indicators Summary")
                report_content.append("\nThe technical analysis includes the following indicators for each company:")
                report_content.append("- **Moving Averages:** 20-day, 50-day, and 200-day Simple Moving Averages")
                report_content.append("- **RSI (Relative Strength Index):** Momentum oscillator")
                report_content.append("- **MACD:** Moving Average Convergence Divergence")
                report_content.append("- **Bollinger Bands:** Volatility indicators")
                report_content.append(f"\nTechnical data points analyzed: **{len(self.technical_indicators)}**")
            
            # Investment Recommendations section removed - not producing meaningful content
            
            # Risk Warnings
            report_content.append("\n## ⚠️ Risk Warnings")
            report_content.append("\n### Important Disclaimers")
            report_content.append("- This analysis is for informational purposes only")
            report_content.append("- Past performance does not guarantee future results")
            report_content.append("- Always conduct your own research before investing")
            report_content.append("- Consider consulting with a financial advisor")
            report_content.append("- Market conditions can change rapidly")
            
            # Data Sources
            report_content.append("\n## 📚 Data Sources")
            report_content.append("\n- **Stock Data:** Yahoo Finance (yfinance)")
            report_content.append("- **Financial Statements:** Company filings via yfinance")
            report_content.append("- **Analysis Period:** Maximum available historical data")
            report_content.append(f"- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")
            
            # Footer
            report_content.append("\n---")
            report_content.append(f"\n*Report generated by Unified Investment Analysis Pipeline*")
            report_content.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
            # Write report to file
            report_filename = f"reports/investment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            print(f"✅ Comprehensive markdown report generated: {report_filename}")
            print(f"📄 Report contains {len(report_content)} lines of analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Failed to generate markdown report: {e}")
            logger.error(f"Markdown report generation error: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete unified pipeline"""
        print("🚀 Starting Unified Investment Analysis Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Collect all raw data
            if not self.collect_all_data():
                print("❌ ERROR: Failed to collect data. Exiting.")
                return False
            
            # Step 2: Perform technical analysis
            self.perform_technical_analysis()
            
            # Step 3: Perform risk analysis
            self.perform_risk_analysis()
            
            # Step 4: Perform portfolio analysis
            self.perform_portfolio_analysis()
            
            # Step 5: Export all results to PowerBI
            if not self.export_powerbi_data():
                print("❌ ERROR: Failed to export PowerBI data.")
                return False
            
            # Step 6: Generate comprehensive markdown report
            if not self.generate_markdown_report():
                print("❌ ERROR: Failed to generate markdown report.")
                return False
            
            print("\n" + "=" * 60)
            print("UNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Generate summary
            self.generate_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Pipeline failed: {e}")
            logger.error(f"Pipeline error: {traceback.format_exc()}")
            return False
    
    def generate_summary(self):
        """Generate summary of all analysis results"""
        try:
            print("\n📊 ANALYSIS SUMMARY")
            print("=" * 40)
            
            if self.company_summaries:
                print(f"\n📊 Analyzed {len(self.company_summaries)} companies:")
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
                        print(f"    Error displaying summary for {ticker}: {e}")
            
            print(f"\n📁 Output files created:")
            print(f"  • powerbi/data/ - CSV files for PowerBI dashboard")
            print(f"  • Technical indicators: {len(self.technical_indicators)} records")
            print(f"  • Risk metrics: {len(self.risk_metrics)} companies")
            print(f"  • Portfolio analysis: {len(self.portfolio_analysis)} methods")
            
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 PowerBI data ready in 'powerbi/data/' directory")
            
        except Exception as e:
            print(f"❌ ERROR: Failed to generate summary: {e}")
            logger.error(f"Summary generation error: {e}")

def main():
    """Main function that runs the unified pipeline"""
    print("🚀 Starting Unified Renewable Energy Investment Analysis Pipeline")
    print("=" * 80)
    print("🎯 Zero duplication • PowerBI export at the end • All analysis results")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues and try again.")
        return False
    
    try:
        print(f"\n{'='*60}")
        print("INITIALIZING UNIFIED PIPELINE")
        print(f"{'='*60}")
        
        # Create pipeline instance
        pipeline = UnifiedInvestmentPipeline()
        
        print(f"\n{'='*60}")
        print("RUNNING COMPLETE ANALYSIS PIPELINE")
        print(f"{'='*60}")
        
        # Run the complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            print(f"\n{'='*60}")
            print("🎉 UNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print("\n📊 Analysis Results Available:")
            print("• 'powerbi/data/' - Complete PowerBI dataset (6 CSV files)")
            print("  - company_summary.csv - Company fundamentals and metrics")
            print("  - stock_prices.csv - Historical price data")
            print("  - risk_metrics.csv - Risk analysis results")
            print("  - technical_indicators.csv - Technical indicators")
            print("  - portfolio_analysis.csv - Portfolio optimization results")
            print("  - correlation_matrix.csv - Correlation analysis")
            print("\n📄 'reports/' - Comprehensive markdown analysis reports")
            print("  - investment_analysis_report_YYYYMMDD_HHMMSS.md")
            print("\n📈 Next Steps:")
            print("1. Open PowerBI Desktop")
            print("2. Import CSV files from 'powerbi/data/' directory")
            print("3. Follow the PowerBI setup guide in 'powerbi/' directory")
            print("4. Create your interactive dashboard!")
            print("5. Review the markdown report in 'reports/' directory")
            
            return True
        else:
            print(f"\n{'='*60}")
            print("❌ UNIFIED PIPELINE FAILED")
            print(f"{'='*60}")
            print("Please check the logs and fix any issues before continuing.")
            return False
            
    except ImportError as e:
        print(f"❌ ERROR: Failed to import unified pipeline: {e}")
        print("Please ensure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ ERROR: Pipeline failed with unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 