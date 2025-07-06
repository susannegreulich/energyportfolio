"""
Presentation Script with Pre-computed Results for Renewable Energy Investment Analysis
This script is designed to be called from the main pipeline with results from previous steps
"""

import sys
import logging
import os
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PresentationManager:
    """Manages presentation generation with optional pre-computed results"""
    
    def __init__(self, analysis_results=None, optimization_results=None):
        """Initialize presentation manager with optional pre-computed results"""
        self.analysis_results = analysis_results
        self.optimization_results = optimization_results
        
        # Import required modules
        try:
            from investment_analysis import InvestmentAnalyzer
            from database import DatabaseManager
            self.analyzer = InvestmentAnalyzer()
            self.db_manager = DatabaseManager()
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        
        # Create output directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('charts', exist_ok=True)
        
        logger.info("PresentationManager initialized successfully")
    
    def generate_comprehensive_presentation(self):
        """Generate comprehensive presentation with all analysis results"""
        try:
            logger.info("Starting comprehensive presentation generation...")
            print("Starting comprehensive presentation generation...")
            
            # Generate analysis reports if not provided
            if self.analysis_results is None:
                logger.info("No pre-computed analysis results, running analysis...")
                print("Running investment analysis...")
                analysis_results, detailed_reports = self.analyzer.analyze_all_companies()
                self.analysis_results = analysis_results
            else:
                logger.info("Using pre-computed analysis results")
                print("Using pre-computed analysis results")
            
            # Generate optimization results if not provided
            if self.optimization_results is None:
                logger.info("No pre-computed optimization results, running optimization...")
                print("Running portfolio optimization...")
                from portfolio_optimization import PortfolioOptimizer
                optimizer = PortfolioOptimizer()
                self.optimization_results = optimizer.generate_portfolio_optimization()
            else:
                logger.info("Using pre-computed optimization results")
                print("Using pre-computed optimization results")
            
            # Generate summary report
            self._generate_summary_report()
            
            # Generate detailed reports
            self._generate_detailed_reports()
            
            # Generate portfolio report
            if self.optimization_results:
                self._generate_portfolio_report()
            
            logger.info("Comprehensive presentation generation completed!")
            print("Comprehensive presentation generation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error generating comprehensive presentation: {e}")
            print(f"ERROR: Failed to generate comprehensive presentation: {e}")
            return False
    
    def _generate_summary_report(self):
        """Generate summary report with key findings"""
        try:
            report_path = "reports/investment_summary_report.txt"
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("RENEWABLE ENERGY INVESTMENT ANALYSIS - SUMMARY REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if self.analysis_results:
                    f.write("INVESTMENT RECOMMENDATIONS:\n")
                    f.write("-" * 40 + "\n")
                    
                    buy_count = 0
                    hold_count = 0
                    sell_count = 0
                    
                    # Handle list of results
                    if isinstance(self.analysis_results, list):
                        for result in self.analysis_results:
                            if isinstance(result, dict) and 'recommendation' in result:
                                ticker = result.get('ticker', 'Unknown')
                                recommendation = result['recommendation']
                                f.write(f"{ticker}: {recommendation}\n")
                                
                                if 'BUY' in recommendation.upper():
                                    buy_count += 1
                                elif 'SELL' in recommendation.upper():
                                    sell_count += 1
                                else:
                                    hold_count += 1
                    # Handle dictionary of results (fallback)
                    elif isinstance(self.analysis_results, dict):
                        for ticker, result in self.analysis_results.items():
                            if isinstance(result, dict) and 'recommendation' in result:
                                recommendation = result['recommendation']
                                f.write(f"{ticker}: {recommendation}\n")
                                
                                if 'BUY' in recommendation.upper():
                                    buy_count += 1
                                elif 'SELL' in recommendation.upper():
                                    sell_count += 1
                                else:
                                    hold_count += 1
                    
                    f.write(f"\nSummary: {buy_count} BUY, {hold_count} HOLD, {sell_count} SELL\n\n")
                
                if self.optimization_results:
                    f.write("PORTFOLIO OPTIMIZATION RESULTS:\n")
                    f.write("-" * 40 + "\n")
                    
                    for method, result in self.optimization_results.items():
                        f.write(f"{method.upper()} OPTIMIZATION:\n")
                        f.write(f"  Expected Return: {result.get('expected_return', 0):.4f} ({result.get('expected_return', 0)*100:.2f}%)\n")
                        f.write(f"  Volatility: {result.get('volatility', 0):.4f} ({result.get('volatility', 0)*100:.2f}%)\n")
                        f.write(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF SUMMARY REPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Summary report generated: {report_path}")
            print(f"Summary report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            print(f"ERROR: Failed to generate summary report: {e}")
    
    def _generate_detailed_reports(self):
        """Generate detailed reports for each company"""
        try:
            if not self.analysis_results:
                logger.warning("No analysis results available for detailed reports")
                return
            
            # Handle list of results
            if isinstance(self.analysis_results, list):
                for result in self.analysis_results:
                    if isinstance(result, dict):
                        ticker = result.get('ticker', 'Unknown')
                        report_path = f"reports/{ticker}_detailed_report.txt"
                        
                        with open(report_path, 'w') as f:
                            f.write("=" * 80 + "\n")
                            f.write(f"DETAILED ANALYSIS REPORT: {ticker}\n")
                            f.write("=" * 80 + "\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            
                            # Write all available data
                            for key, value in result.items():
                                f.write(f"{key.upper()}:\n")
                                f.write("-" * 20 + "\n")
                                f.write(f"{value}\n\n")
                        
                        logger.info(f"Detailed report generated for {ticker}: {report_path}")
                
                print(f"Detailed reports generated for {len(self.analysis_results)} companies")
            
            # Handle dictionary of results (fallback)
            elif isinstance(self.analysis_results, dict):
                for ticker, result in self.analysis_results.items():
                    if isinstance(result, dict):
                        report_path = f"reports/{ticker}_detailed_report.txt"
                        
                        with open(report_path, 'w') as f:
                            f.write("=" * 80 + "\n")
                            f.write(f"DETAILED ANALYSIS REPORT: {ticker}\n")
                            f.write("=" * 80 + "\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            
                            # Write all available data
                            for key, value in result.items():
                                f.write(f"{key.upper()}:\n")
                                f.write("-" * 20 + "\n")
                                f.write(f"{value}\n\n")
                        
                        logger.info(f"Detailed report generated for {ticker}: {report_path}")
                
                print(f"Detailed reports generated for {len(self.analysis_results)} companies")
            
        except Exception as e:
            logger.error(f"Error generating detailed reports: {e}")
            print(f"ERROR: Failed to generate detailed reports: {e}")
    
    def _generate_portfolio_report(self):
        """Generate portfolio optimization report"""
        try:
            if not self.optimization_results:
                logger.warning("No optimization results available for portfolio report")
                return
            
            report_path = "reports/portfolio_optimization_report.txt"
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("PORTFOLIO OPTIMIZATION REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for method, result in self.optimization_results.items():
                    f.write(f"{method.upper()} OPTIMIZATION:\n")
                    f.write("-" * 40 + "\n")
                    
                    for key, value in result.items():
                        if key == 'weights':
                            f.write(f"{key}:\n")
                            # Handle different weight formats
                            if isinstance(value, dict):
                                for asset, weight in value.items():
                                    f.write(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)\n")
                            elif hasattr(value, '__iter__') and not isinstance(value, str):
                                # Handle numpy array or list
                                for i, weight in enumerate(value):
                                    f.write(f"  Asset {i+1}: {weight:.4f} ({weight*100:.2f}%)\n")
                            else:
                                f.write(f"  {value}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF PORTFOLIO REPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Portfolio report generated: {report_path}")
            print(f"Portfolio report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating portfolio report: {e}")
            print(f"ERROR: Failed to generate portfolio report: {e}")

def create_presentation_manager(analysis_results=None, optimization_results=None):
    """Create and return a PresentationManager instance"""
    return PresentationManager(analysis_results, optimization_results)

def load_results_from_files():
    """Load pre-computed results from files if they exist"""
    analysis_results = None
    optimization_results = None
    
    # Try to load analysis results
    analysis_file = "temp_analysis_results.pkl"
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'rb') as f:
                analysis_results = pickle.load(f)
            logger.info("Loaded pre-computed analysis results")
        except Exception as e:
            logger.warning(f"Could not load analysis results: {e}")
    
    # Try to load optimization results
    optimization_file = "temp_optimization_results.pkl"
    if os.path.exists(optimization_file):
        try:
            with open(optimization_file, 'rb') as f:
                optimization_results = pickle.load(f)
            logger.info("Loaded pre-computed optimization results")
        except Exception as e:
            logger.warning(f"Could not load optimization results: {e}")
    
    return analysis_results, optimization_results

def cleanup_temp_files():
    """Clean up temporary result files"""
    temp_files = ["temp_analysis_results.pkl", "temp_optimization_results.pkl"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Cleaned up temporary file: {file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {file}: {e}")

def main():
    """Main function for presentation with pre-computed results"""
    try:
        logger.info("Starting presentation generation with pre-computed results...")
        print("Starting presentation generation with pre-computed results...")
        
        # Load pre-computed results
        analysis_results, optimization_results = load_results_from_files()
        
        if analysis_results is None and optimization_results is None:
            print("WARNING: No pre-computed results found. Running full analysis...")
        
        # Create presentation manager with pre-computed results
        presentation_manager = create_presentation_manager(analysis_results, optimization_results)
        
        # Generate comprehensive presentation
        success = presentation_manager.generate_comprehensive_presentation()
        
        if success:
            # Clean up temporary files
            cleanup_temp_files()
            
            logger.info("Presentation generation with pre-computed results completed!")
            print("Presentation generation with pre-computed results completed successfully!")
            return True
        else:
            logger.error("Presentation generation failed")
            print("FAILURE: Presentation generation failed")
            return False
        
    except Exception as e:
        logger.error(f"Critical error in presentation with results: {e}")
        print(f"CRITICAL ERROR: {e}")
        print("Please check the logs for more details.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 