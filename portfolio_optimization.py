"""
Portfolio Optimization Script for Renewable Energy Investment Analysis
Handles portfolio optimization analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
warnings.filterwarnings('ignore')

from config import *
from database import DatabaseManager
from investment_analysis import InvestmentAnalyzer

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Handles portfolio optimization analysis"""
    
    def __init__(self):
        """Initialize the portfolio optimizer"""
        self.db_manager = DatabaseManager()
        self.analyzer = InvestmentAnalyzer()
        
        # Create charts directory
        os.makedirs('charts', exist_ok=True)
    
    def generate_portfolio_optimization(self, save_path=None):
        """Generate portfolio optimization analysis"""
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, skipping portfolio optimization")
                return None
            
            # Get returns data for all companies
            all_returns = {}
            
            for name, ticker in self.analyzer.companies.items():
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    returns = prices['Close'].pct_change().dropna()
                    all_returns[ticker] = returns
            
            if len(all_returns) < 2:
                logger.warning("Insufficient data for portfolio optimization")
                return None
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                logger.warning("No valid returns data for portfolio optimization")
                return None
            
            # Perform portfolio optimization
            optimization_results = {}
            
            for method in ['sharpe', 'min_variance', 'max_return']:
                result = self.analyzer.portfolio_optimization(returns_df, method)
                if result:
                    optimization_results[method] = result
            
            # Create visualization
            if optimization_results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Portfolio Optimization Analysis', fontsize=16)
                
                # Risk-return scatter plot
                methods = list(optimization_results.keys())
                returns = [optimization_results[m]['expected_return'] for m in methods]
                volatilities = [optimization_results[m]['volatility'] for m in methods]
                
                axes[0, 0].scatter(volatilities, returns, s=100, alpha=0.7)
                for i, method in enumerate(methods):
                    axes[0, 0].annotate(method.title(), (volatilities[i], returns[i]), 
                                      xytext=(5, 5), textcoords='offset points')
                axes[0, 0].set_xlabel('Volatility')
                axes[0, 0].set_ylabel('Expected Return')
                axes[0, 0].set_title('Risk-Return Profile')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Portfolio weights comparison
                x = np.arange(len(returns_df.columns))
                width = 0.25
                
                for i, method in enumerate(methods):
                    weights = optimization_results[method]['weights']
                    axes[0, 1].bar(x + i*width, weights, width, label=method.title(), alpha=0.7)
                
                axes[0, 1].set_xlabel('Assets')
                axes[0, 1].set_ylabel('Weight')
                axes[0, 1].set_title('Portfolio Weights by Method')
                axes[0, 1].set_xticks(x + width)
                axes[0, 1].set_xticklabels(returns_df.columns, rotation=45)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Sharpe ratio comparison
                sharpe_ratios = [optimization_results[m]['sharpe_ratio'] for m in methods]
                
                axes[1, 0].bar(methods, sharpe_ratios, color='lightgreen', alpha=0.7)
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

def main():
    """Main function for portfolio optimization"""
    logger.info("Starting portfolio optimization...")
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer()
    
    # Generate portfolio optimization
    portfolio_chart_path = "charts/portfolio_optimization.png"
    optimization_results = optimizer.generate_portfolio_optimization(portfolio_chart_path)
    
    if optimization_results:
        logger.info("Portfolio optimization completed successfully!")
        print("Portfolio optimization completed successfully!")
        print("Charts saved to: charts/")
        
        # Print summary of results
        print("\nPortfolio Optimization Results:")
        print("-" * 40)
        for method, result in optimization_results.items():
            print(f"{method.upper()} OPTIMIZATION:")
            print(f"  Expected Return: {result['expected_return']:.4f} ({result['expected_return']*100:.2f}%)")
            print(f"  Volatility: {result['volatility']:.4f} ({result['volatility']*100:.2f}%)")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print()
    else:
        logger.warning("Portfolio optimization failed or returned no results")
        print("Portfolio optimization failed or returned no results")

if __name__ == "__main__":
    main() 