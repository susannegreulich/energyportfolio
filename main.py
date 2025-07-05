"""
Main Investment Analysis Script for Renewable Energy Companies
Orchestrates the complete analysis pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
import warnings
warnings.filterwarnings('ignore')

from config import *
from database import DatabaseManager, create_database
from enhanced_fetch_data import EnhancedDataFetcher
from investment_analysis import InvestmentAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenewableEnergyInvestmentAnalysis:
    """Main class for renewable energy investment analysis"""
    
    def __init__(self):
        """Initialize the analysis system"""
        self.db_manager = DatabaseManager()
        self.data_fetcher = EnhancedDataFetcher()
        self.analyzer = InvestmentAnalyzer()
        
        # Create output directories
        os.makedirs('analysis', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('charts', exist_ok=True)
        
    def setup_database(self):
        """Set up database and load initial data"""
        logger.info("Setting up database...")
        
        try:
            # Create database and tables
            if not create_database():
                logger.warning("Failed to create database, continuing without database")
                return True  # Continue without database
            
            # Load data
            logger.info("Loading data into database...")
            if not self.data_fetcher.load_data_to_database():
                logger.warning("Failed to load data into database, continuing without database")
                return True  # Continue without database
            
            return True
        except Exception as e:
            logger.warning(f"Database setup failed: {e}, continuing without database")
            return True  # Continue without database
    
    def generate_technical_charts(self, ticker, save_path=None):
        """Generate technical analysis charts"""
        try:
            if not self.db_manager.connect():
                logger.warning(f"Database connection failed for {ticker}, skipping chart generation")
                return
            
            # Get stock data
            prices = self.db_manager.get_stock_prices(ticker)
            if prices.empty:
                logger.warning(f"No data available for {ticker}")
                return
            
            # Calculate technical indicators
            technical = self.analyzer.technical_analysis(prices)
            
            # Create subplots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle(f'Technical Analysis: {ticker}', fontsize=16)
            
            # Price and moving averages
            axes[0].plot(prices.index, prices['Close'], label='Close Price', linewidth=2)
            axes[0].plot(prices.index, technical['SMA_20'], label='SMA 20', alpha=0.7)
            axes[0].plot(prices.index, technical['SMA_50'], label='SMA 50', alpha=0.7)
            axes[0].plot(prices.index, technical['SMA_200'], label='SMA 200', alpha=0.7)
            axes[0].fill_between(prices.index, technical['BB_Upper'], technical['BB_Lower'], 
                               alpha=0.1, label='Bollinger Bands')
            axes[0].set_title('Price and Moving Averages')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # RSI
            axes[1].plot(prices.index, technical['RSI'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[1].set_title('Relative Strength Index (RSI)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # MACD
            axes[2].plot(prices.index, technical['MACD'], label='MACD', color='blue')
            axes[2].plot(prices.index, technical['MACD_Signal'], label='Signal', color='red')
            axes[2].bar(prices.index, technical['MACD_Histogram'], label='Histogram', alpha=0.5)
            axes[2].set_title('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Technical chart saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating technical charts for {ticker}: {e}")
        finally:
            self.db_manager.disconnect()
    
    def generate_risk_analysis_charts(self, save_path=None):
        """Generate risk analysis charts for all companies"""
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, skipping risk analysis charts")
                return
            
            # Get data for all companies
            all_returns = {}
            risk_metrics = {}
            
            for name, ticker in self.analyzer.companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
                    risk_metrics[ticker] = self.analyzer.risk_analysis(returns)
            
            if not all_returns:
                logger.warning("No data available for risk analysis")
                return
            
            # Create risk analysis charts
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Risk Analysis: Renewable Energy Companies', fontsize=16)
            
            # Volatility comparison
            volatilities = [metrics['Volatility'] for metrics in risk_metrics.values()]
            tickers = list(risk_metrics.keys())
            
            axes[0, 0].bar(tickers, volatilities, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Annualized Volatility')
            axes[0, 0].set_ylabel('Volatility')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe ratio comparison
            sharpe_ratios = [metrics['Sharpe_Ratio'] for metrics in risk_metrics.values()]
            
            axes[0, 1].bar(tickers, sharpe_ratios, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Sharpe Ratio')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Maximum drawdown comparison
            max_drawdowns = [metrics['Max_Drawdown'] for metrics in risk_metrics.values()]
            
            axes[1, 0].bar(tickers, max_drawdowns, color='salmon', alpha=0.7)
            axes[1, 0].set_title('Maximum Drawdown')
            axes[1, 0].set_ylabel('Max Drawdown')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Correlation heatmap
            returns_df = pd.DataFrame(all_returns)
            correlation_matrix = returns_df.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
            axes[1, 1].set_title('Correlation Matrix')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Risk analysis chart saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating risk analysis charts: {e}")
        finally:
            self.db_manager.disconnect()
    
    def generate_portfolio_optimization(self, save_path=None):
        """Generate portfolio optimization analysis"""
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, skipping portfolio optimization")
                return
            
            # Get returns data for all companies
            all_returns = {}
            
            for name, ticker in self.analyzer.companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
            
            if len(all_returns) < 2:
                logger.warning("Insufficient data for portfolio optimization")
                return
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("No valid returns data for portfolio optimization")
                return
            
            # Perform portfolio optimization
            optimization_results = {}
            
            for method in ['sharpe', 'min_variance', 'max_return']:
                result = self.analyzer.portfolio_optimization(returns_df, method)
                if result:
                    optimization_results[method] = result
            
            if not optimization_results:
                logger.warning("Portfolio optimization failed")
                return
            
            # Create optimization comparison chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Portfolio Optimization Analysis', fontsize=16)
            
            # Portfolio weights comparison
            methods = list(optimization_results.keys())
            tickers = list(returns_df.columns)
            
            for i, method in enumerate(methods):
                weights = optimization_results[method]['weights']
                axes[0, 0].bar([f"{ticker}\n({method})" for ticker in tickers], 
                              weights, alpha=0.7, label=method)
            
            axes[0, 0].set_title('Portfolio Weights by Optimization Method')
            axes[0, 0].set_ylabel('Weight')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Risk-return comparison
            returns_opt = []
            risks_opt = []
            sharpe_opt = []
            
            for method in methods:
                result = optimization_results[method]
                returns_opt.append(result['expected_return'])
                risks_opt.append(result['volatility'])
                sharpe_opt.append(result['sharpe_ratio'])
            
            axes[0, 1].scatter(risks_opt, returns_opt, s=100, alpha=0.7)
            for i, method in enumerate(methods):
                axes[0, 1].annotate(method, (risks_opt[i], returns_opt[i]), 
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[0, 1].set_xlabel('Portfolio Volatility')
            axes[0, 1].set_ylabel('Expected Return')
            axes[0, 1].set_title('Risk-Return Profile')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Sharpe ratio comparison
            axes[1, 0].bar(methods, sharpe_opt, color='lightblue', alpha=0.7)
            axes[1, 0].set_title('Sharpe Ratio by Optimization Method')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Backtest results for Sharpe-optimized portfolio
            if 'sharpe' in optimization_results:
                sharpe_weights = optimization_results['sharpe']['weights']
                backtest = self.analyzer.backtest_portfolio(returns_df, sharpe_weights)
                
                axes[1, 1].plot(backtest['portfolio_values'].index, 
                               backtest['portfolio_values'].values, 
                               label='Portfolio Value', linewidth=2)
                axes[1, 1].set_title('Portfolio Performance (Sharpe Optimized)')
                axes[1, 1].set_ylabel('Portfolio Value ($)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Portfolio optimization chart saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error generating portfolio optimization: {e}")
            return None
        finally:
            self.db_manager.disconnect()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive investment analysis report"""
        logger.info("Generating comprehensive analysis report...")
        
        # Create report file
        report_path = f"reports/renewable_energy_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RENEWABLE ENERGY INVESTMENT ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: {DATA_START_DATE} to {DATA_END_DATE}\n")
            f.write("="*80 + "\n\n")
            
            # Company analysis
            f.write("1. INDIVIDUAL COMPANY ANALYSIS\n")
            f.write("-" * 40 + "\n\n")
            
            analysis_results = self.analyzer.analyze_all_companies()
            
            if analysis_results:
                # Summary table
                f.write("Investment Recommendations Summary:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Ticker':<12} {'Score':<8} {'Recommendation':<15} {'Sharpe':<8} {'Volatility':<12}\n")
                f.write("-" * 50 + "\n")
                
                for result in analysis_results:
                    ticker = result['ticker']
                    score = result['score']
                    recommendation = result['recommendation']
                    sharpe = result['risk_metrics']['Sharpe_Ratio']
                    volatility = result['risk_metrics']['Volatility']
                    
                    f.write(f"{ticker:<12} {score:<8.1f} {recommendation:<15} {sharpe:<8.3f} {volatility:<12.3f}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("2. PORTFOLIO OPTIMIZATION RESULTS\n")
                f.write("="*80 + "\n\n")
                
                # Portfolio optimization
                optimization_results = self.generate_portfolio_optimization()
                
                if optimization_results:
                    f.write("Portfolio Optimization Results:\n")
                    f.write("-" * 40 + "\n")
                    
                    for method, result in optimization_results.items():
                        f.write(f"\n{method.upper()} OPTIMIZATION:\n")
                        f.write(f"  Expected Return: {result['expected_return']:.4f} ({result['expected_return']*100:.2f}%)\n")
                        f.write(f"  Volatility: {result['volatility']:.4f} ({result['volatility']*100:.2f}%)\n")
                        f.write(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}\n")
                        f.write("  Portfolio Weights:\n")
                        
                        for i, ticker in enumerate(self.analyzer.companies.values()):
                            if i < len(result['weights']):
                                weight = result['weights'][i]
                                f.write(f"    {ticker}: {weight:.4f} ({weight*100:.2f}%)\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("3. RISK ANALYSIS SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                # Risk analysis summary
                f.write("Key Risk Metrics:\n")
                f.write("-" * 30 + "\n")
                
                for result in analysis_results:
                    ticker = result['ticker']
                    risk_metrics = result['risk_metrics']
                    
                    f.write(f"\n{ticker}:\n")
                    f.write(f"  Volatility: {risk_metrics['Volatility']:.4f}\n")
                    f.write(f"  Sharpe Ratio: {risk_metrics['Sharpe_Ratio']:.4f}\n")
                    f.write(f"  Max Drawdown: {risk_metrics['Max_Drawdown']:.4f}\n")
                    f.write(f"  VaR (95%): {risk_metrics['VaR_95']:.4f}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("4. INVESTMENT RECOMMENDATIONS\n")
                f.write("="*80 + "\n\n")
                
                # Top recommendations
                top_picks = [r for r in analysis_results if r['score'] >= 2]
                if top_picks:
                    f.write("TOP RECOMMENDATIONS:\n")
                    f.write("-" * 25 + "\n")
                    for result in top_picks:
                        f.write(f"• {result['ticker']}: {result['recommendation']}\n")
                        f.write(f"  Score: {result['score']}, Reasons: {', '.join(result['reasons'])}\n\n")
                else:
                    f.write("No strong buy recommendations at this time.\n\n")
                
                # Risk warnings
                f.write("RISK WARNINGS:\n")
                f.write("-" * 15 + "\n")
                f.write("• Past performance does not guarantee future results\n")
                f.write("• Renewable energy sector is subject to regulatory changes\n")
                f.write("• Currency fluctuations may affect international stocks\n")
                f.write("• Consider diversification across different renewable energy subsectors\n")
                f.write("• Monitor policy changes and government incentives\n\n")
                
                f.write("="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
        
        logger.info(f"Comprehensive report saved to {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete investment analysis pipeline"""
        logger.info("Starting complete renewable energy investment analysis...")
        
        # Step 1: Setup database and load data
        logger.info("Step 1: Setting up database and loading data...")
        if not self.setup_database():
            logger.error("Failed to setup database. Exiting.")
            return False
        
        # Step 2: Generate technical charts
        logger.info("Step 2: Generating technical analysis charts...")
        for name, ticker in self.analyzer.companies.items():
            chart_path = f"charts/technical_{ticker.replace('.', '_')}.png"
            self.generate_technical_charts(ticker, chart_path)
        
        # Step 3: Generate risk analysis charts
        logger.info("Step 3: Generating risk analysis charts...")
        risk_chart_path = "charts/risk_analysis.png"
        self.generate_risk_analysis_charts(risk_chart_path)
        
        # Step 4: Generate portfolio optimization
        logger.info("Step 4: Generating portfolio optimization...")
        portfolio_chart_path = "charts/portfolio_optimization.png"
        self.generate_portfolio_optimization(portfolio_chart_path)
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating comprehensive report...")
        report_path = self.generate_comprehensive_report()
        
        logger.info("Complete analysis finished successfully!")
        logger.info(f"Report saved to: {report_path}")
        logger.info("Charts saved to: charts/")
        
        return True

def main():
    """Main function"""
    print("Renewable Energy Investment Analysis System")
    print("="*50)
    
    # Initialize analysis system
    analysis_system = RenewableEnergyInvestmentAnalysis()
    
    # Run complete analysis
    success = analysis_system.run_complete_analysis()
    
    if success:
        print("\nAnalysis completed successfully!")
        print("Check the 'reports/' and 'charts/' directories for results.")
    else:
        print("\nAnalysis failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
