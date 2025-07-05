"""
Comprehensive Presentation Module for Renewable Energy Investment Analysis
Handles both report generation and visualization charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from config import *
from database import DatabaseManager
from investment_analysis import InvestmentAnalyzer
from portfolio_optimization import PortfolioOptimizer

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PresentationManager:
    """Handles comprehensive presentation generation including reports and charts"""
    
    def __init__(self):
        """Initialize the presentation manager"""
        self.db_manager = DatabaseManager()
        self.analyzer = InvestmentAnalyzer()
        self.optimizer = PortfolioOptimizer()
        
        # Create directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('charts', exist_ok=True)
    
    # ==================== REPORT GENERATION METHODS ====================
    
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
            
            analysis_results, detailed_reports = self.analyzer.analyze_all_companies()
            
            # Write detailed individual company reports
            if detailed_reports:
                f.write("DETAILED INDIVIDUAL COMPANY ANALYSIS\n")
                f.write("=" * 50 + "\n\n")
                for report in detailed_reports:
                    f.write(report)
                    f.write("\n" + "="*80 + "\n\n")
            
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
                optimization_results = self.optimizer.generate_portfolio_optimization()
                
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
    
    # ==================== VISUALIZATION METHODS ====================
    
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
    
    def generate_technical_charts(self, ticker, save_path=None):
        """Generate technical analysis charts for a single company"""
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
    
    def generate_all_technical_charts(self):
        """Generate technical charts for all companies"""
        logger.info("Generating technical analysis charts for all companies...")
        
        for name, ticker in self.analyzer.companies.items():
            logger.info(f"Generating technical chart for {ticker}...")
            chart_path = f"charts/technical_{ticker.replace('.', '_')}.png"
            self.generate_technical_charts(ticker, chart_path)
    
    def generate_comprehensive_dashboard(self, save_path=None):
        """Generate a comprehensive dashboard with both risk and technical analysis"""
        try:
            if not self.db_manager.connect():
                logger.warning("Database connection failed, skipping dashboard generation")
                return
            
            # Create a large figure for the dashboard
            fig = plt.figure(figsize=(20, 16))
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
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
                logger.warning("No data available for dashboard")
                return
            
            tickers = list(risk_metrics.keys())
            
            # 1. Volatility comparison (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            volatilities = [metrics['Volatility'] for metrics in risk_metrics.values()]
            ax1.bar(tickers, volatilities, color='skyblue', alpha=0.7)
            ax1.set_title('Annualized Volatility', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 2. Sharpe ratio comparison (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            sharpe_ratios = [metrics['Sharpe_Ratio'] for metrics in risk_metrics.values()]
            ax2.bar(tickers, sharpe_ratios, color='lightgreen', alpha=0.7)
            ax2.set_title('Sharpe Ratio', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. Maximum drawdown comparison (second row left)
            ax3 = fig.add_subplot(gs[1, 0])
            max_drawdowns = [metrics['Max_Drawdown'] for metrics in risk_metrics.values()]
            ax3.bar(tickers, max_drawdowns, color='salmon', alpha=0.7)
            ax3.set_title('Maximum Drawdown', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 4. Correlation heatmap (second row right)
            ax4 = fig.add_subplot(gs[1, 1])
            returns_df = pd.DataFrame(all_returns)
            correlation_matrix = returns_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=ax4, cbar_kws={'label': 'Correlation'})
            ax4.set_title('Correlation Matrix', fontsize=12)
            
            # 5. Price comparison for top 3 companies (bottom half)
            ax5 = fig.add_subplot(gs[2:, :])
            
            # Get top 3 companies by Sharpe ratio
            top_companies = sorted(risk_metrics.items(), key=lambda x: x[1]['Sharpe_Ratio'], reverse=True)[:3]
            
            for ticker, metrics in top_companies:
                prices = self.db_manager.get_stock_prices(ticker)
                if not prices.empty:
                    # Normalize prices to start at 100 for comparison
                    normalized_prices = (prices['Close'] / prices['Close'].iloc[0]) * 100
                    ax5.plot(prices.index, normalized_prices, label=f'{ticker} (Sharpe: {metrics["Sharpe_Ratio"]:.2f})', linewidth=2)
            
            ax5.set_title('Price Performance Comparison (Top 3 by Sharpe Ratio)', fontsize=14)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_ylabel('Normalized Price (Base=100)')
            
            fig.suptitle('Renewable Energy Investment Analysis Dashboard', fontsize=16, y=0.98)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Comprehensive dashboard saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating comprehensive dashboard: {e}")
        finally:
            self.db_manager.disconnect()
    
    def generate_all_charts(self):
        """Generate all types of charts"""
        logger.info("Starting comprehensive chart generation...")
        
        # Generate risk analysis charts
        risk_chart_path = "charts/risk_analysis.png"
        self.generate_risk_analysis_charts(risk_chart_path)
        
        # Generate technical charts for all companies
        self.generate_all_technical_charts()
        
        # Generate comprehensive dashboard
        dashboard_path = "charts/comprehensive_dashboard.png"
        self.generate_comprehensive_dashboard(dashboard_path)
        
        logger.info("All charts generation completed!")
    
    # ==================== COMPREHENSIVE PRESENTATION METHODS ====================
    
    def generate_comprehensive_presentation(self):
        """Generate both comprehensive report and all charts"""
        logger.info("Starting comprehensive presentation generation...")
        
        # Generate comprehensive report
        report_path = self.generate_comprehensive_report()
        
        # Generate all charts
        self.generate_all_charts()
        
        logger.info("Comprehensive presentation generation completed!")
        print("Comprehensive presentation generation completed!")
        print(f"Report saved to: {report_path}")
        print("Charts saved to: charts/")
        print("Generated files:")
        print("- charts/risk_analysis.png")
        print("- charts/technical_[TICKER].png (for each company)")
        print("- charts/comprehensive_dashboard.png")
        
        return report_path

def main():
    """Main function for comprehensive presentation generation"""
    logger.info("Starting comprehensive presentation generation...")
    
    # Initialize presentation manager
    presentation_manager = PresentationManager()
    
    # Generate comprehensive presentation
    presentation_manager.generate_comprehensive_presentation()
    
    logger.info("Comprehensive presentation generation completed!")

if __name__ == "__main__":
    main() 