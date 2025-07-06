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
import sys
import traceback
warnings.filterwarnings('ignore')

# Set up logging to show INFO level messages on console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    from config import *
    from database import DatabaseManager
    from investment_analysis import InvestmentAnalyzer
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure all required files (config.py, database.py, investment_analysis.py) exist in the same directory.")
    sys.exit(1)

class PortfolioOptimizer:
    """Handles portfolio optimization analysis"""
    
    def __init__(self):
        """Initialize the portfolio optimizer"""
        try:
            self.db_manager = DatabaseManager()
            self.analyzer = InvestmentAnalyzer()
            
            # Create charts directory
            os.makedirs('charts', exist_ok=True)
            logger.info("PortfolioOptimizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PortfolioOptimizer: {e}")
            print(f"ERROR: Failed to initialize PortfolioOptimizer: {e}")
            raise
    
    def generate_portfolio_optimization(self, save_path=None):
        """Generate portfolio optimization analysis"""
        try:
            logger.info("Starting portfolio optimization analysis...")
            
            if not self.db_manager.connect():
                error_msg = "Database connection failed, skipping portfolio optimization"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return None
            
            # Get returns data for all companies
            all_returns = {}
            logger.info("Fetching stock price data...")
            
            for name, ticker in self.analyzer.companies.items():
                try:
                    prices = self.db_manager.get_stock_prices(ticker)
                    if not prices.empty:
                        returns = prices['Close'].pct_change().dropna()
                        all_returns[ticker] = returns
                        logger.info(f"Successfully processed data for {ticker}")
                    else:
                        logger.warning(f"No price data available for {ticker}")
                except Exception as e:
                    logger.error(f"Error processing data for {ticker}: {e}")
                    print(f"WARNING: Error processing data for {ticker}: {e}")
            
            if len(all_returns) < 2:
                error_msg = "Insufficient data for portfolio optimization (need at least 2 assets)"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return None
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(all_returns)
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                error_msg = "No valid returns data for portfolio optimization after cleaning"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return None
            
            logger.info(f"Created returns DataFrame with {len(returns_df)} rows and {len(returns_df.columns)} assets")
            
            # Perform portfolio optimization
            optimization_results = {}
            logger.info("Performing portfolio optimization...")
            
            for method in ['sharpe', 'min_variance', 'max_return']:
                try:
                    result = self.analyzer.portfolio_optimization(returns_df, method)
                    if result:
                        optimization_results[method] = result
                        logger.info(f"Successfully completed {method} optimization")
                    else:
                        logger.warning(f"{method} optimization returned no results")
                except Exception as e:
                    logger.error(f"Error in {method} optimization: {e}")
                    print(f"ERROR: {method} optimization failed: {e}")
            
            if not optimization_results:
                error_msg = "No optimization methods produced valid results"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return None
            
            # Create visualization
            logger.info("Creating portfolio optimization visualization...")
            try:
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
                    try:
                        sharpe_weights = optimization_results['sharpe']['weights']
                        backtest = self.analyzer.backtest_portfolio(returns_df, sharpe_weights)
                        
                        axes[1, 1].plot(backtest['portfolio_values'].index, 
                                       backtest['portfolio_values'].values, 
                                       label='Portfolio Value', linewidth=2)
                        axes[1, 1].set_title('Portfolio Performance (Sharpe Optimized)')
                        axes[1, 1].set_ylabel('Portfolio Value ($)')
                        axes[1, 1].legend()
                        axes[1, 1].grid(True, alpha=0.3)
                    except Exception as e:
                        logger.error(f"Error creating backtest plot: {e}")
                        print(f"WARNING: Could not create backtest plot: {e}")
                        axes[1, 1].text(0.5, 0.5, 'Backtest data unavailable', 
                                       ha='center', va='center', transform=axes[1, 1].transAxes)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Portfolio optimization chart saved to {save_path}")
                    print(f"SUCCESS: Portfolio optimization chart saved to {save_path}")
                else:
                    plt.show()
                
                plt.close()
                
                return optimization_results
                
            except Exception as e:
                logger.error(f"Error creating visualization: {e}")
                print(f"ERROR: Failed to create visualization: {e}")
                return optimization_results  # Return results even if visualization fails
            
        except Exception as e:
            logger.error(f"Error generating portfolio optimization: {e}")
            print(f"ERROR: Failed to generate portfolio optimization: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            try:
                self.db_manager.disconnect()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")

def main():
    """Main function for portfolio optimization"""
    try:
        logger.info("Starting portfolio optimization...")
        print("Starting portfolio optimization...")
        
        # Initialize portfolio optimizer
        optimizer = PortfolioOptimizer()
        
        # Generate portfolio optimization
        portfolio_chart_path = "charts/portfolio_optimization.png"
        optimization_results = optimizer.generate_portfolio_optimization(portfolio_chart_path)
        
        if optimization_results:
            logger.info("Portfolio optimization completed successfully!")
            print("SUCCESS: Portfolio optimization completed successfully!")
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
            
            # Save results for use by other scripts
            try:
                import pickle
                with open("temp_optimization_results.pkl", 'wb') as f:
                    pickle.dump(optimization_results, f)
                print("Optimization results saved for use by other scripts")
            except Exception as e:
                print(f"WARNING: Could not save optimization results: {e}")
        else:
            logger.warning("Portfolio optimization failed or returned no results")
            print("FAILURE: Portfolio optimization failed or returned no results")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"CRITICAL ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 