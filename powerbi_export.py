"""
PowerBI Data Export Script for Renewable Energy Investment Analysis
Exports analysis data to CSV files for PowerBI dashboard creation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import yfinance as yf
from config import *
from database import DatabaseManager
from investment_analysis import InvestmentAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowerBIExporter:
    """Export data for PowerBI dashboard"""
    
    def __init__(self):
        """Initialize the exporter"""
        self.db_manager = DatabaseManager()
        self.analyzer = InvestmentAnalyzer()
        
        # Create PowerBI data directory
        os.makedirs('powerbi/data', exist_ok=True)
        
        # Filter out delisted companies
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
        self.valid_companies = {}
        for name, ticker in companies_dict.items():
            if ticker != 'SGRE.MC':  # Remove delisted company
                self.valid_companies[name] = ticker
        
    def export_company_summary(self):
        """Export company summary data"""
        logger.info("Exporting company summary data...")
        
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, generating summary from yfinance data")
                return self._export_company_summary_from_yfinance()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, generating summary from yfinance data")
            return self._export_company_summary_from_yfinance()
        
        try:
            # Get company summary
            summary = self.db_manager.get_company_summary()
            
            if not summary.empty:
                # Add additional calculated fields
                summary['Market_Cap_Billions'] = summary['market_cap'] / 1e9 if 'market_cap' in summary.columns else None
                summary['PE_Category'] = pd.cut(summary['pe_ratio'], 
                                              bins=[0, 15, 25, 50, float('inf')], 
                                              labels=['Value', 'Fair Value', 'Growth', 'High Growth'])
                summary['Risk_Category'] = pd.cut(summary['volatility'], 
                                                bins=[0, 0.2, 0.4, 0.6, float('inf')], 
                                                labels=['Low', 'Medium', 'High', 'Very High'])
                
                # Export to CSV
                output_path = 'powerbi/data/company_summary.csv'
                summary.to_csv(output_path, index=False)
                logger.info(f"Company summary exported to {output_path}")
                return True
            else:
                logger.warning("No company summary data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting company summary: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def _export_company_summary_from_yfinance(self):
        """Export company summary data using yfinance when database is unavailable"""
        try:
            summary_data = []
            
            for name, ticker in self.valid_companies.items():
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Get latest price
                    hist = stock.history(period='1d')
                    current_price = hist['Close'].iloc[-1] if not hist.empty else None
                    
                    summary_data.append({
                        'company_id': len(summary_data) + 1,
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
                        'volatility': None,  # Will be calculated separately
                        'beta': info.get('beta'),
                        'sharpe_ratio': None  # Will be calculated separately
                    })
                except Exception as e:
                    logger.warning(f"Error fetching data for {ticker}: {e}")
                    continue
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Add calculated fields
                summary_df['Market_Cap_Billions'] = summary_df['market_cap'] / 1e9 if 'market_cap' in summary_df.columns else None
                summary_df['PE_Category'] = pd.cut(summary_df['pe_ratio'], 
                                                  bins=[0, 15, 25, 50, float('inf')], 
                                                  labels=['Value', 'Fair Value', 'Growth', 'High Growth'])
                
                # Export to CSV
                output_path = 'powerbi/data/company_summary.csv'
                summary_df.to_csv(output_path, index=False)
                logger.info(f"Company summary exported to {output_path}")
                return True
            else:
                logger.warning("No company summary data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting company summary from yfinance: {e}")
            return False
    
    def export_stock_prices(self):
        """Export stock price data for all companies"""
        logger.info("Exporting stock price data...")
        
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, generating stock prices from yfinance data")
                return self._export_stock_prices_from_yfinance()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, generating stock prices from yfinance data")
            return self._export_stock_prices_from_yfinance()
        
        try:
            all_prices = []
            
            for name, ticker in self.valid_companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    prices['Ticker'] = ticker
                    prices['Company_Name'] = name
                    prices.reset_index(inplace=True)
                    all_prices.append(prices)
            
            if all_prices:
                # Combine all price data
                combined_prices = pd.concat(all_prices, ignore_index=True)
                
                # Add calculated fields
                combined_prices['Year'] = combined_prices['date'].dt.year
                combined_prices['Month'] = combined_prices['date'].dt.month
                combined_prices['Quarter'] = combined_prices['date'].dt.quarter
                combined_prices['Day_of_Week'] = combined_prices['date'].dt.dayofweek
                
                # Calculate returns
                combined_prices['Daily_Return'] = combined_prices.groupby('Ticker')['Close'].pct_change()
                # Calculate cumulative returns properly
                combined_prices['Cumulative_Return'] = combined_prices.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod())
                
                # Export to CSV
                output_path = 'powerbi/data/stock_prices.csv'
                combined_prices.to_csv(output_path, index=False)
                logger.info(f"Stock prices exported to {output_path}")
                return True
            else:
                logger.warning("No stock price data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting stock prices: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def _export_stock_prices_from_yfinance(self):
        """Export stock price data using yfinance when database is unavailable"""
        try:
            all_prices = []
            
            for name, ticker in self.valid_companies.items():
                try:
                    stock = yf.Ticker(ticker)
                    prices = stock.history(start=DATA_START_DATE, end=DATA_END_DATE)
                    
                    if not prices.empty:
                        prices['Ticker'] = ticker
                        prices['Company_Name'] = name
                        prices.reset_index(inplace=True)
                        all_prices.append(prices)
                except Exception as e:
                    logger.warning(f"Error fetching stock prices for {ticker}: {e}")
                    continue
            
            if all_prices:
                # Combine all price data
                combined_prices = pd.concat(all_prices, ignore_index=True)
                
                # Add calculated fields
                combined_prices['Year'] = pd.to_datetime(combined_prices['Date'], utc=True).dt.year
                combined_prices['Month'] = pd.to_datetime(combined_prices['Date'], utc=True).dt.month
                combined_prices['Quarter'] = pd.to_datetime(combined_prices['Date'], utc=True).dt.quarter
                combined_prices['Day_of_Week'] = pd.to_datetime(combined_prices['Date'], utc=True).dt.dayofweek
                
                # Calculate returns
                combined_prices['Daily_Return'] = combined_prices.groupby('Ticker')['Close'].pct_change()
                # Calculate cumulative returns properly
                combined_prices['Cumulative_Return'] = combined_prices.groupby('Ticker')['Daily_Return'].transform(lambda x: (1 + x).cumprod())
                
                # Export to CSV
                output_path = 'powerbi/data/stock_prices.csv'
                combined_prices.to_csv(output_path, index=False)
                logger.info(f"Stock prices exported to {output_path}")
                return True
            else:
                logger.warning("No stock price data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting stock prices from yfinance: {e}")
            return False
    
    def export_risk_metrics(self):
        """Export risk metrics data"""
        logger.info("Exporting risk metrics data...")
        
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, generating risk metrics from yfinance data")
                return self._export_risk_metrics_from_yfinance()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, generating risk metrics from yfinance data")
            return self._export_risk_metrics_from_yfinance()
        
        try:
            all_risk_data = []
            
            for name, ticker in self.valid_companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    returns = prices['Close'].pct_change().dropna()
                    risk_metrics = self.analyzer.risk_analysis(returns)
                    
                    # Add company info
                    risk_metrics['Ticker'] = ticker
                    risk_metrics['Company_Name'] = name
                    risk_metrics['Analysis_Date'] = datetime.now().date()
                    
                    all_risk_data.append(risk_metrics)
            
            if all_risk_data:
                # Create DataFrame
                risk_df = pd.DataFrame(all_risk_data)
                
                # Add risk categories
                risk_df['Risk_Level'] = pd.cut(risk_df['Volatility'], 
                                             bins=[0, 0.2, 0.4, 0.6, float('inf')], 
                                             labels=['Low', 'Medium', 'High', 'Very High'])
                risk_df['Sharpe_Category'] = pd.cut(risk_df['Sharpe_Ratio'], 
                                                   bins=[float('-inf'), 0, 0.5, 1, float('inf')], 
                                                   labels=['Poor', 'Fair', 'Good', 'Excellent'])
                
                # Export to CSV
                output_path = 'powerbi/data/risk_metrics.csv'
                risk_df.to_csv(output_path, index=False)
                logger.info(f"Risk metrics exported to {output_path}")
                return True
            else:
                logger.warning("No risk metrics data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting risk metrics: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def _export_risk_metrics_from_yfinance(self):
        """Export risk metrics data using yfinance when database is unavailable"""
        try:
            all_risk_data = []
            
            for name, ticker in self.valid_companies.items():
                try:
                    stock = yf.Ticker(ticker)
                    prices = stock.history(start=DATA_START_DATE, end=DATA_END_DATE)
                    
                    if not prices.empty:
                        returns = prices['Close'].pct_change().dropna()
                        risk_metrics = self.analyzer.risk_analysis(returns)
                        
                        # Add company info
                        risk_metrics['Ticker'] = ticker
                        risk_metrics['Company_Name'] = name
                        risk_metrics['Analysis_Date'] = datetime.now().date()
                        
                        all_risk_data.append(risk_metrics)
                except Exception as e:
                    logger.warning(f"Error calculating risk metrics for {ticker}: {e}")
                    continue
            
            if all_risk_data:
                # Create DataFrame
                risk_df = pd.DataFrame(all_risk_data)
                
                # Add risk categories
                risk_df['Risk_Level'] = pd.cut(risk_df['Volatility'], 
                                             bins=[0, 0.2, 0.4, 0.6, float('inf')], 
                                             labels=['Low', 'Medium', 'High', 'Very High'])
                risk_df['Sharpe_Category'] = pd.cut(risk_df['Sharpe_Ratio'], 
                                                   bins=[float('-inf'), 0, 0.5, 1, float('inf')], 
                                                   labels=['Poor', 'Fair', 'Good', 'Excellent'])
                
                # Export to CSV
                output_path = 'powerbi/data/risk_metrics.csv'
                risk_df.to_csv(output_path, index=False)
                logger.info(f"Risk metrics exported to {output_path}")
                return True
            else:
                logger.warning("No risk metrics data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting risk metrics from yfinance: {e}")
            return False
    
    def _export_portfolio_analysis_from_yfinance(self):
        """Export portfolio analysis data using yfinance when database is unavailable"""
        try:
            # Get returns data for all companies
            all_returns = {}
            
            for name, ticker in self.valid_companies.items():
                try:
                    stock = yf.Ticker(ticker)
                    prices = stock.history(start=DATA_START_DATE, end=DATA_END_DATE)
                    
                    if not prices.empty:
                        returns = prices['Close'].pct_change().dropna()
                        all_returns[ticker] = returns
                except Exception as e:
                    logger.warning(f"Error fetching data for {ticker}: {e}")
                    continue
            
            if len(all_returns) < 2:
                logger.warning("Insufficient data for portfolio analysis")
                return False
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("No valid returns data for portfolio analysis")
                return False
            
            # Perform portfolio optimization
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
                    # Create portfolio data
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
                # Create DataFrame
                portfolio_df = pd.DataFrame(portfolio_data)
                
                # Export to CSV
                output_path = 'powerbi/data/portfolio_analysis.csv'
                portfolio_df.to_csv(output_path, index=False)
                logger.info(f"Portfolio analysis exported to {output_path}")
                return True
            else:
                logger.warning("No portfolio analysis data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting portfolio analysis from yfinance: {e}")
            return False
    
    def export_portfolio_analysis(self):
        """Export portfolio analysis data"""
        logger.info("Exporting portfolio analysis data...")
        
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, generating portfolio analysis from yfinance data")
                return self._export_portfolio_analysis_from_yfinance()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, generating portfolio analysis from yfinance data")
            return self._export_portfolio_analysis_from_yfinance()
        
        try:
            # Get returns data for all companies
            all_returns = {}
            
            for name, ticker in self.valid_companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
            
            if len(all_returns) < 2:
                logger.warning("Insufficient data for portfolio analysis")
                return False
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("No valid returns data for portfolio analysis")
                return False
            
            # Perform portfolio optimization
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
                    # Create portfolio data
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
                # Create DataFrame
                portfolio_df = pd.DataFrame(portfolio_data)
                
                # Export to CSV
                output_path = 'powerbi/data/portfolio_analysis.csv'
                portfolio_df.to_csv(output_path, index=False)
                logger.info(f"Portfolio analysis exported to {output_path}")
                return True
            else:
                logger.warning("No portfolio analysis data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting portfolio analysis: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def export_technical_indicators(self):
        """Export technical indicators data"""
        logger.info("Exporting technical indicators data...")
        
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, generating technical indicators from yfinance data")
                return self._export_technical_indicators_from_yfinance()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, generating technical indicators from yfinance data")
            return self._export_technical_indicators_from_yfinance()
        
        try:
            all_technical_data = []
            
            for name, ticker in self.valid_companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    # Calculate technical indicators
                    technical = self.analyzer.technical_analysis(prices)
                    
                    # Create DataFrame with technical data
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
                # Combine all technical data
                combined_technical = pd.concat(all_technical_data, ignore_index=True)
                
                # Export to CSV
                output_path = 'powerbi/data/technical_indicators.csv'
                combined_technical.to_csv(output_path, index=False)
                logger.info(f"Technical indicators exported to {output_path}")
                return True
            else:
                logger.warning("No technical indicators data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting technical indicators: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def _export_technical_indicators_from_yfinance(self):
        """Export technical indicators data using yfinance when database is unavailable"""
        try:
            all_technical_data = []
            
            for name, ticker in self.valid_companies.items():
                try:
                    stock = yf.Ticker(ticker)
                    prices = stock.history(start=DATA_START_DATE, end=DATA_END_DATE)
                    
                    if not prices.empty:
                        # Calculate technical indicators
                        technical = self.analyzer.technical_analysis(prices)
                        
                        # Create DataFrame with technical data
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
                    logger.warning(f"Error calculating technical indicators for {ticker}: {e}")
                    continue
            
            if all_technical_data:
                # Combine all technical data
                combined_technical = pd.concat(all_technical_data, ignore_index=True)
                
                # Export to CSV
                output_path = 'powerbi/data/technical_indicators.csv'
                combined_technical.to_csv(output_path, index=False)
                logger.info(f"Technical indicators exported to {output_path}")
                return True
            else:
                logger.warning("No technical indicators data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting technical indicators from yfinance: {e}")
            return False
    
    def export_correlation_matrix(self):
        """Export correlation matrix data"""
        logger.info("Exporting correlation matrix data...")
        
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, generating correlation matrix from yfinance data")
                return self._export_correlation_matrix_from_yfinance()
        except Exception as e:
            logger.warning(f"Database connection failed: {e}, generating correlation matrix from yfinance data")
            return self._export_correlation_matrix_from_yfinance()
        
        try:
            # Get returns data for all companies
            all_returns = {}
            
            for name, ticker in self.valid_companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
            
            if len(all_returns) < 2:
                logger.warning("Insufficient data for correlation analysis")
                return False
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("No valid returns data for correlation analysis")
                return False
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Convert to long format for PowerBI
            correlation_data = []
            for ticker1 in correlation_matrix.index:
                for ticker2 in correlation_matrix.columns:
                    correlation_data.append({
                        'Ticker1': ticker1,
                        'Ticker2': ticker2,
                        'Correlation': correlation_matrix.loc[ticker1, ticker2]
                    })
            
            # Create DataFrame
            correlation_df = pd.DataFrame(correlation_data)
            
            # Export to CSV
            output_path = 'powerbi/data/correlation_matrix.csv'
            correlation_df.to_csv(output_path, index=False)
            logger.info(f"Correlation matrix exported to {output_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error exporting correlation matrix: {e}")
            return False
        finally:
            self.db_manager.disconnect()
    
    def _export_correlation_matrix_from_yfinance(self):
        """Export correlation matrix data using yfinance when database is unavailable"""
        try:
            # Get returns data for all companies
            all_returns = {}
            
            for name, ticker in self.valid_companies.items():
                try:
                    stock = yf.Ticker(ticker)
                    prices = stock.history(start=DATA_START_DATE, end=DATA_END_DATE)
                    
                    if not prices.empty:
                        returns = prices['Close'].pct_change().dropna()
                        all_returns[ticker] = returns
                except Exception as e:
                    logger.warning(f"Error fetching data for {ticker}: {e}")
                    continue
            
            if len(all_returns) < 2:
                logger.warning("Insufficient data for correlation analysis")
                return False
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("No valid returns data for correlation analysis")
                return False
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Convert to long format for PowerBI
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
                # Create DataFrame
                correlation_df = pd.DataFrame(correlation_data)
                
                # Export to CSV
                output_path = 'powerbi/data/correlation_matrix.csv'
                correlation_df.to_csv(output_path, index=False)
                logger.info(f"Correlation matrix exported to {output_path}")
                return True
            else:
                logger.warning("No correlation matrix data available")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting correlation matrix from yfinance: {e}")
            return False
    
    def export_all_data(self):
        """Export all data for PowerBI"""
        logger.info("Starting PowerBI data export...")
        
        success = True
        
        # Export all data types
        if not self.export_company_summary():
            success = False
        
        if not self.export_stock_prices():
            success = False
        
        if not self.export_risk_metrics():
            success = False
        
        if not self.export_portfolio_analysis():
            success = False
        
        if not self.export_technical_indicators():
            success = False
        
        if not self.export_correlation_matrix():
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
            print("\nYou can now import these files into PowerBI to create your dashboard.")
        else:
            logger.error("Some data exports failed. Check the logs for details.")
        
        return success

def main():
    """Main function"""
    print("PowerBI Data Export for Renewable Energy Investment Analysis")
    print("="*60)
    
    exporter = PowerBIExporter()
    success = exporter.export_all_data()
    
    if success:
        print("\nPowerBI data export completed successfully!")
    else:
        print("\nPowerBI data export failed. Please check the logs for details.")

if __name__ == "__main__":
    main() 