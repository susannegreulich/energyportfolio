"""
Comprehensive Investment Analysis Module for Renewable Energy Companies
Includes Technical, Fundamental, Risk Analysis, and Portfolio Optimization
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import *
from database import DatabaseManager

class InvestmentAnalyzer:
    """Comprehensive investment analysis for renewable energy companies"""
    
    def __init__(self):
        """Initialize the analyzer"""
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
        
    def calculate_returns(self, prices):
        """Calculate daily and cumulative returns"""
        returns = prices.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        return returns, cumulative_returns
    
    def calculate_volatility(self, returns, window=252):
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_beta(self, stock_returns, market_returns, window=252):
        """Calculate rolling beta"""
        beta = stock_returns.rolling(window=window).cov(market_returns) / market_returns.rolling(window=window).var()
        return beta
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02, window=252):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        sharpe = excess_returns.rolling(window=window).mean() / returns.rolling(window=window).std()
        return sharpe * np.sqrt(252)
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def calculate_var(self, returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def technical_analysis(self, prices):
        """Perform technical analysis"""
        # Simple Moving Averages
        sma_20 = prices['Close'].rolling(window=20).mean()
        sma_50 = prices['Close'].rolling(window=50).mean()
        sma_200 = prices['Close'].rolling(window=200).mean()
        
        # RSI
        delta = prices['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices['Close'].ewm(span=12).mean()
        ema_26 = prices['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        # Bollinger Bands
        bb_middle = prices['Close'].rolling(window=20).mean()
        bb_std = prices['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        return {
            'SMA_20': sma_20,
            'SMA_50': sma_50,
            'SMA_200': sma_200,
            'RSI': rsi,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Histogram': macd_histogram,
            'BB_Upper': bb_upper,
            'BB_Middle': bb_middle,
            'BB_Lower': bb_lower
        }
    
    def fundamental_analysis(self, ticker):
        """Perform fundamental analysis using available data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key metrics
            metrics = {
                'PE_Ratio': info.get('trailingPE'),
                'PB_Ratio': info.get('priceToBook'),
                'PS_Ratio': info.get('priceToSalesTrailing12Months'),
                'Debt_to_Equity': info.get('debtToEquity'),
                'Current_Ratio': info.get('currentRatio'),
                'ROE': info.get('returnOnEquity'),
                'ROA': info.get('returnOnAssets'),
                'Profit_Margin': info.get('profitMargins'),
                'Revenue_Growth': info.get('revenueGrowth'),
                'Earnings_Growth': info.get('earningsGrowth'),
                'Dividend_Yield': info.get('dividendYield'),
                'Payout_Ratio': info.get('payoutRatio')
            }
            
            return metrics
        except Exception as e:
            print(f"Error fetching fundamental data for {ticker}: {e}")
            return {}
    
    def risk_analysis(self, returns, benchmark_returns=None):
        """Perform comprehensive risk analysis"""
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)
        annual_return = returns.mean() * 252
        sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        
        # Value at Risk
        var_95 = self.calculate_var(returns, 0.05)
        cvar_95 = self.calculate_cvar(returns, 0.05)
        
        # Beta calculation if benchmark provided
        beta = None
        if benchmark_returns is not None:
            beta = returns.cov(benchmark_returns) / benchmark_returns.var()
        
        return {
            'Volatility': volatility,
            'Annual_Return': annual_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Beta': beta
        }
    
    def correlation_analysis(self, returns_data):
        """Analyze correlations between stocks"""
        correlation_matrix = returns_data.corr()
        return correlation_matrix
    
    def portfolio_optimization(self, returns_data, method='sharpe'):
        """Optimize portfolio weights using different methods"""
        n_assets = len(returns_data.columns)
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        if method == 'sharpe':
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
                return -sharpe  # Minimize negative Sharpe ratio
            
        elif method == 'min_variance':
            # Minimize variance
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_vol
        
        elif method == 'max_return':
            # Maximize return
            def objective(weights):
                portfolio_return = np.sum(expected_returns * weights)
                return -portfolio_return  # Minimize negative return
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio
            }
        else:
            print("Optimization failed")
            return None
    
    def backtest_portfolio(self, returns_data, weights, initial_investment=100000):
        """Backtest portfolio performance"""
        # Calculate portfolio returns
        portfolio_returns = returns_data.dot(weights)
        
        # Calculate portfolio value over time
        portfolio_values = initial_investment * (1 + portfolio_returns).cumprod()
        
        # Calculate performance metrics
        total_return = (portfolio_values.iloc[-1] - initial_investment) / initial_investment
        annual_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_values / initial_investment)
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def generate_analysis_report(self, ticker, start_date=None, end_date=None):
        """Generate comprehensive analysis report for a company"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS REPORT: {ticker}")
        print(f"{'='*60}")
        
        # Get stock data - try database first, fallback to yfinance
        prices = None
        try:
            if self.db_manager.connect():
                prices = self.db_manager.get_stock_prices(ticker, start_date, end_date)
                self.db_manager.disconnect()
        except Exception as e:
            print(f"Database connection failed, using yfinance: {e}")
        
        if prices is None or prices.empty:
            # Fallback to yfinance
            try:
                stock = yf.Ticker(ticker)
                prices = stock.history(start=start_date or DATA_START_DATE, end=end_date or DATA_END_DATE)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                return None
        
        if prices.empty:
            print(f"No data available for {ticker}")
            return None
        
        # Calculate returns
        returns, cumulative_returns = self.calculate_returns(prices['Close'])
        
        # 1. Technical Analysis
        print("\n1. TECHNICAL ANALYSIS")
        print("-" * 30)
        technical = self.technical_analysis(prices)
        
        current_price = prices['Close'].iloc[-1]
        sma_20_current = technical['SMA_20'].iloc[-1]
        sma_50_current = technical['SMA_50'].iloc[-1]
        rsi_current = technical['RSI'].iloc[-1]
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"20-day SMA: ${sma_20_current:.2f}")
        print(f"50-day SMA: ${sma_50_current:.2f}")
        print(f"RSI: {rsi_current:.2f}")
        
        # Technical signals
        if current_price > sma_20_current > sma_50_current:
            print("Technical Signal: BULLISH (Price above both SMAs)")
        elif current_price < sma_20_current < sma_50_current:
            print("Technical Signal: BEARISH (Price below both SMAs)")
        else:
            print("Technical Signal: NEUTRAL")
        
        if rsi_current > 70:
            print("RSI Signal: OVERBOUGHT")
        elif rsi_current < 30:
            print("RSI Signal: OVERSOLD")
        else:
            print("RSI Signal: NEUTRAL")
        
        # 2. Risk Analysis
        print("\n2. RISK ANALYSIS")
        print("-" * 30)
        risk_metrics = self.risk_analysis(returns)
        
        for metric, value in risk_metrics.items():
            if value is not None:
                if 'Ratio' in metric or 'Return' in metric:
                    print(f"{metric}: {value:.4f}")
                elif 'Drawdown' in metric:
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.6f}")
        
        # 3. Fundamental Analysis
        print("\n3. FUNDAMENTAL ANALYSIS")
        print("-" * 30)
        fundamental = self.fundamental_analysis(ticker)
        
        if fundamental:
            for metric, value in fundamental.items():
                if value is not None:
                    if 'Yield' in metric or 'Growth' in metric or 'Ratio' in metric:
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value:.2f}")
        
        # 4. Investment Recommendation
        print("\n4. INVESTMENT RECOMMENDATION")
        print("-" * 30)
        
        score = 0
        reasons = []
        
        # Technical score
        if current_price > sma_20_current > sma_50_current:
            score += 2
            reasons.append("Strong technical trend")
        elif current_price < sma_20_current < sma_50_current:
            score -= 1
            reasons.append("Weak technical trend")
        
        if 30 <= rsi_current <= 70:
            score += 1
            reasons.append("RSI in normal range")
        elif rsi_current < 30:
            score += 1
            reasons.append("Oversold conditions")
        elif rsi_current > 70:
            score -= 1
            reasons.append("Overbought conditions")
        
        # Risk score
        if risk_metrics['Sharpe_Ratio'] > 1.0:
            score += 2
            reasons.append("Strong risk-adjusted returns")
        elif risk_metrics['Sharpe_Ratio'] > 0.5:
            score += 1
            reasons.append("Good risk-adjusted returns")
        elif risk_metrics['Sharpe_Ratio'] < 0:
            score -= 1
            reasons.append("Poor risk-adjusted returns")
        
        if risk_metrics['Max_Drawdown'] > -0.2:
            score += 1
            reasons.append("Acceptable drawdown")
        else:
            score -= 1
            reasons.append("High drawdown risk")
        
        # Fundamental score
        if fundamental.get('PE_Ratio') and fundamental['PE_Ratio'] < PE_RATIO_THRESHOLD:
            score += 1
            reasons.append("Attractive P/E ratio")
        
        if fundamental.get('ROE') and fundamental['ROE'] > ROE_THRESHOLD:
            score += 1
            reasons.append("Strong ROE")
        
        # Generate recommendation
        if score >= 4:
            recommendation = "STRONG BUY"
        elif score >= 2:
            recommendation = "BUY"
        elif score >= 0:
            recommendation = "HOLD"
        elif score >= -2:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"
        
        print(f"Overall Score: {score}")
        print(f"Recommendation: {recommendation}")
        print("Key Factors:")
        for reason in reasons:
            print(f"  • {reason}")
        
        return {
            'ticker': ticker,
            'score': score,
            'recommendation': recommendation,
            'reasons': reasons,
            'technical': technical,
            'risk_metrics': risk_metrics,
            'fundamental': fundamental
        }
    
    def analyze_all_companies(self):
        """Analyze all companies and generate comparative report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RENEWABLE ENERGY INVESTMENT ANALYSIS")
        print(f"{'='*80}")
        
        results = []
        
        for name, ticker in self.companies.items():
            print(f"\nAnalyzing {name} ({ticker})...")
            result = self.generate_analysis_report(ticker)
            if result:
                results.append(result)
        
        # Generate comparative analysis
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        if results:
            # Create summary DataFrame
            summary_data = []
            for result in results:
                summary_data.append({
                    'Ticker': result['ticker'],
                    'Score': result['score'],
                    'Recommendation': result['recommendation'],
                    'Sharpe_Ratio': result['risk_metrics']['Sharpe_Ratio'],
                    'Volatility': result['risk_metrics']['Volatility'],
                    'Max_Drawdown': result['risk_metrics']['Max_Drawdown']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Score', ascending=False)
            
            print("\nRanked by Investment Score:")
            print(summary_df.to_string(index=False))
            
            # Top recommendations
            print(f"\nTOP RECOMMENDATIONS:")
            top_picks = summary_df[summary_df['Score'] >= 2]
            if not top_picks.empty:
                for _, row in top_picks.iterrows():
                    print(f"  • {row['Ticker']}: {row['Recommendation']} (Score: {row['Score']})")
            else:
                print("  No strong buy recommendations at this time.")
        
        return results

if __name__ == "__main__":
    analyzer = InvestmentAnalyzer()
    analyzer.analyze_all_companies() 