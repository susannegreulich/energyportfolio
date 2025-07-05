# Renewable Energy Investment Analysis Project

A comprehensive investment analysis system for the top 10 renewable energy companies, built with Python, SQL, and PowerBI. This project follows standard investment analysis protocols and provides detailed technical, fundamental, and risk analysis.

## 🎯 Project Overview

This project analyzes the following top renewable energy companies:
- **Orsted** (ORSTED.CO) - Danish offshore wind leader
- **Vestas** (VWS.CO) - Wind turbine manufacturer
- **NextEra Energy** (NEE) - US renewable energy utility
- **Iberdrola** (IBE.MC) - Spanish energy company
- **Enel** (ENEL.MI) - Italian energy company
- **Brookfield Renewable** (BEP) - Global renewable energy operator
- **EDP Renovaveis** (EDPR.LS) - Portuguese renewable energy
- **Siemens Gamesa** (SGRE.MC) - Wind turbine manufacturer
- **Plug Power** (PLUG) - Hydrogen fuel cell technology
- **First Solar** (FSLR) - Solar panel manufacturer

## 📊 Analysis Components

### 1. Technical Analysis
- **Moving Averages**: 20, 50, and 200-day Simple Moving Averages
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **Price patterns and trend analysis**

### 2. Fundamental Analysis
- **Valuation Ratios**: P/E, P/B, P/S ratios
- **Financial Health**: Debt-to-Equity, Current Ratio
- **Profitability**: ROE, ROA, Profit Margins
- **Growth Metrics**: Revenue and Earnings Growth
- **Dividend Analysis**: Yield and Payout Ratios

### 3. Risk Analysis
- **Volatility**: Annualized standard deviation
- **Beta**: Market correlation measure
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst historical decline
- **Value at Risk (VaR)**: Potential loss estimation
- **Conditional VaR (CVaR)**: Expected shortfall

### 4. Portfolio Optimization
- **Sharpe Ratio Optimization**: Maximize risk-adjusted returns
- **Minimum Variance**: Minimize portfolio risk
- **Maximum Return**: Maximize expected returns
- **Equal Weight**: Benchmark comparison
- **Backtesting**: Historical performance simulation

## 🏗️ Project Structure

```
energy_investment_project/
├── config.py                    # Configuration settings
├── database.py                 # Database connection and operations
├── fetch_data.py      # Enhanced data fetching with database integration
├── investment_analysis.py      # Core investment analysis engine
├── main.py                     # Unified analysis orchestration script (NEW)
├── powerbi_export.py           # Legacy PowerBI data export utilities
├── test_unified_analysis.py    # Test script for unified system (NEW)
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── sql/
│   └── create_tables.sql      # Database schema
├── data/
│   └── raw/                   # Raw data storage
├── analysis/                  # Analysis outputs
├── reports/                   # Generated reports
├── charts/                    # Generated charts
└── powerbi/
    ├── data/                  # PowerBI data exports (CSV files)
    └── dashboard.pbix         # PowerBI dashboard file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **PostgreSQL** database server (optional - system works without it)
3. **PowerBI Desktop** (for dashboard creation)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd energy_investment_project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the database** (optional):
   - Update `config.py` with your PostgreSQL credentials
   - Ensure PostgreSQL is running
   - System will work without database using direct data collection

4. **Set up API keys** (optional):
   - Get a free Finnhub API key from [finnhub.io](https://finnhub.io/)
   - Update `config.py` with your API key

### Running the Analysis

**NEW: Unified Data Collection System**

The system now uses a unified approach where `fetch_data.py` collects all data once and exports to both CSV (for PowerBI) and database (for main analysis).

1. **Run the complete analysis** (recommended):
   ```bash
   python main.py
   ```
   This will:
   - Call `fetch_data.py` to collect all data once
   - Export CSV files for PowerBI
   - Load data to database for main analysis
   - Generate analysis reports and charts

2. **Run data collection only**:
   ```bash
   python fetch_data.py
   ```
   This will:
   - Collect all data once from Yahoo Finance
   - Export CSV files for PowerBI
   - Load data to database

3. **Test the unified system**:
   ```bash
   python test_unified_analysis.py
   ```

4. **View results**:
   - Check `reports/` for detailed analysis reports
   - Check `charts/` for technical analysis charts
   - Check `powerbi/data/` for PowerBI-ready data files

**Note**: The old `powerbi_export.py` script is still available but the unified `fetch_data.py` is now more efficient.

## 📈 Analysis Workflow

### 1. Data Collection
- **Stock Prices**: Historical daily prices from Yahoo Finance
- **Financial Metrics**: Fundamental ratios and financial statements
- **News Sentiment**: Recent news articles with sentiment analysis
- **Market Data**: Benchmark indices and market indicators

### 2. Data Processing
- **Database Storage**: All data stored in PostgreSQL for efficient querying
- **Data Cleaning**: Handle missing values and outliers
- **Feature Engineering**: Calculate technical indicators and risk metrics

### 3. Analysis Execution
- **Individual Company Analysis**: Technical, fundamental, and risk assessment
- **Portfolio Optimization**: Multi-objective portfolio construction
- **Comparative Analysis**: Cross-company performance comparison

### 4. Reporting
- **Text Reports**: Comprehensive analysis summaries
- **Visual Charts**: Technical analysis and risk visualization
- **PowerBI Integration**: Interactive dashboard creation

## 🗄️ Database Schema

The project uses a comprehensive PostgreSQL database with the following tables:

- **companies**: Company information and metadata
- **stock_prices**: Historical price data
- **financial_metrics**: Fundamental ratios and metrics
- **technical_indicators**: Calculated technical indicators
- **risk_metrics**: Risk analysis results
- **news_articles**: News data with sentiment scores
- **portfolio_performance**: Portfolio backtesting results
- **analysis_results**: Investment recommendations

## 📊 PowerBI Dashboard

The PowerBI dashboard includes:

1. **Company Overview**: Summary metrics and rankings
2. **Technical Analysis**: Interactive charts with indicators
3. **Risk Analysis**: Volatility, correlation, and risk metrics
4. **Portfolio Optimization**: Optimal weights and performance
5. **Comparative Analysis**: Cross-company performance comparison
6. **News Sentiment**: Recent news impact analysis

## 🔧 Configuration

### Database Configuration
Update `config.py` with your database settings:
```python
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "energy_investment_db"
DB_USER = "your_username"
DB_PASSWORD = "your_password"
```

### Analysis Parameters
Customize analysis parameters in `config.py`:
```python
RISK_FREE_RATE = 0.02  # 2% risk-free rate
MARKET_RETURN = 0.10   # 10% expected market return
VOLATILITY_WINDOW = 252  # 1 year for calculations
```

## 📋 API Requirements

### Required APIs
- **Yahoo Finance**: Free stock data (via yfinance)
- **Finnhub** (Optional): News and additional financial data
- **Alpha Vantage** (Optional): Additional financial metrics

### API Setup
1. Get free API keys from respective providers
2. Update `config.py` with your keys
3. The system will work with just Yahoo Finance data

## 📊 Output Files

### Reports
- **Analysis Reports**: Comprehensive investment analysis
- **Risk Reports**: Detailed risk assessment
- **Portfolio Reports**: Optimization results and recommendations

### Charts
- **Technical Charts**: Price, indicators, and patterns
- **Risk Charts**: Volatility, correlation, and drawdown analysis
- **Portfolio Charts**: Optimization results and performance

### PowerBI Data
- **CSV Exports**: All analysis data in PowerBI-ready format
- **Dashboard Template**: Pre-built PowerBI dashboard

## 🎯 Investment Analysis Features

### Technical Analysis
- **Trend Analysis**: Moving averages and price patterns
- **Momentum Indicators**: RSI, MACD, and oscillators
- **Volatility Analysis**: Bollinger Bands and volatility measures
- **Support/Resistance**: Key price levels identification

### Fundamental Analysis
- **Valuation Metrics**: P/E, P/B, P/S ratios
- **Financial Health**: Liquidity and solvency ratios
- **Profitability Analysis**: ROE, ROA, and margins
- **Growth Assessment**: Revenue and earnings trends

### Risk Management
- **Portfolio Risk**: Diversification and correlation analysis
- **Risk Metrics**: VaR, CVaR, and maximum drawdown
- **Stress Testing**: Scenario analysis and sensitivity
- **Risk-Adjusted Returns**: Sharpe and Sortino ratios

## 🔍 Usage Examples

### Individual Company Analysis
```python
from investment_analysis import InvestmentAnalyzer

analyzer = InvestmentAnalyzer()
result = analyzer.generate_analysis_report('NEE')
print(f"Recommendation: {result['recommendation']}")
```

### Portfolio Optimization
```python
from investment_analysis import InvestmentAnalyzer

analyzer = InvestmentAnalyzer()
# Get optimized portfolio weights
optimization = analyzer.portfolio_optimization(returns_data, 'sharpe')
print(f"Optimal weights: {optimization['weights']}")
```

### Database Operations
```python
from database import DatabaseManager

db = DatabaseManager()
db.connect()
summary = db.get_company_summary()
print(summary)
```

## 🚨 Risk Disclaimers

- **Past Performance**: Historical data does not guarantee future results
- **Market Risk**: All investments carry inherent market risk
- **Sector Risk**: Renewable energy sector subject to regulatory changes
- **Currency Risk**: International stocks affected by exchange rates
- **Liquidity Risk**: Some stocks may have limited trading volume

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service.

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the logs for error messages
3. Ensure all dependencies are installed
4. Verify database connectivity

## 🔄 Updates and Maintenance

- **Data Updates**: Run `python fetch_data.py` to update data
- **Analysis Updates**: Run `python main.py` for fresh analysis
- **PowerBI Updates**: Run `python powerbi_export.py` for new data exports

---

**Note**: This is a comprehensive investment analysis tool. Always conduct your own research and consider consulting with financial professionals before making investment decisions.

🎉 Complete Renewable Energy Investment Analysis Project
What I've Built:
📊 Core Analysis Engine (investment_analysis.py)
Technical analysis (RSI, MACD, Bollinger Bands, Moving Averages)
Fundamental analysis (P/E ratios, ROE, debt-to-equity, growth metrics)
Risk analysis (volatility, beta, Sharpe ratio, VaR, maximum drawdown)
Portfolio optimization (Sharpe ratio, minimum variance, maximum return)
Investment scoring and recommendations
️ Database System (database.py + sql/create_tables.sql)
Complete PostgreSQL schema with 8 tables
Optimized queries and indexes
Pre-built views for common analyses
Efficient data management
 Data Pipeline (fetch_data.py)
Multi-source data collection (Yahoo Finance, Finnhub, Alpha Vantage)
Financial statements analysis
News sentiment analysis
Automated database loading
 Main Orchestration (main.py)
End-to-end analysis pipeline
Chart generation and visualization
Comprehensive report generation
Error handling and logging
📊 PowerBI Integration (powerbi_export.py)
CSV exports optimized for PowerBI
Dashboard-ready data formats
Multiple analysis views
⚙️ Configuration & Setup
Centralized configuration (config.py)
Automated setup script (setup.py)
Complete requirements (requirements.txt)
Comprehensive documentation (README.md)
Companies Analyzed:
Orsted (ORSTED.CO) - Danish offshore wind
Vestas (VWS.CO) - Wind turbines
NextEra Energy (NEE) - US renewable utility
Iberdrola (IBE.MC) - Spanish energy
Enel (ENEL.MI) - Italian energy
Brookfield Renewable (BEP) - Global operator
EDP Renovaveis (EDPR.LS) - Portuguese renewable
Siemens Gamesa (SGRE.MC) - Wind turbines
Plug Power (PLUG) - Hydrogen fuel cells
First Solar (FSLR) - Solar panels
Standard Investment Analysis Protocols Followed:
Data Collection - Historical prices, financial statements, news sentiment
Technical Analysis - Moving averages, RSI, MACD, Bollinger Bands
Fundamental Analysis - Valuation ratios, financial health, growth metrics
Risk Analysis - Volatility, correlation, risk-adjusted returns
Portfolio Optimization - Multi-objective optimization strategies
Recommendation System - Comprehensive scoring and buy/hold/sell signals
How to Use:
Setup (already completed): python3 setup.py
Configure Database: Update PostgreSQL credentials in config.py
Run Analysis: python3 main.py
PowerBI Export: python3 powerbi_export.py
View Results: Check reports/, charts/, and powerbi/data/ directories
Output Generated:
Reports: Detailed investment analysis for each company
Charts: Technical analysis and risk visualization
PowerBI Data: CSV files ready for dashboard creation
Database: Complete PostgreSQL database with all analysis data
The project is now complete and ready to use! It follows industry-standard investment analysis protocols and provides a professional-grade system for analyzing renewable energy investments. All dependencies have been installed, the database schema is ready, and the analysis pipeline is fully functional.