# Renewable Energy Investment Analysis Project

A comprehensive investment analysis system for the top 10 renewable energy companies, built with Python and SQL. This project provides detailed technical, fundamental, and risk analysis using a **Unified Pipeline** approach.

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

The project uses a **Unified Investment Analysis Pipeline** that provides comprehensive analysis and exports detailed CSV data and reports.

## 🏗️ Project Structure

```
energy_investment_project/
├── scripts/
│   ├── config.py              # Configuration settings and API keys
│   ├── database.py            # Database connection and operations
│   ├── unified_pipeline.py    # Main unified analysis pipeline
│   ├── setup.py               # Setup and configuration script
│   └── create_tables.sql      # Database schema
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── unified_pipeline.log      # Pipeline execution logs
├── data/                     # Raw stock price data for each company
├── results/                  # Analysis outputs (CSV files and reports)
│   ├── company_summary.csv
│   ├── stock_prices.csv
│   ├── risk_metrics.csv
│   ├── technical_indicators.csv
│   └── investment_analysis_report_YYYYMMDD_HHMMSS.md
```

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

## 📊 Analysis Data Files Generated

### **1. company_summary.csv**
**Contains**:
- Company fundamentals (P/E, ROE, debt-to-equity)
- Market data (price, market cap)
- Risk metrics (volatility, Sharpe ratio)
- Calculated fields (market cap billions, risk categories)

### **2. stock_prices.csv**
**Contains**:
- Historical OHLCV data
- Calculated returns (daily, cumulative)
- Time dimensions (year, month, quarter)

### **3. risk_metrics.csv**
**Contains**:
- Volatility, annual return, Sharpe ratio
- Maximum drawdown, VaR, CVaR
- Risk levels and categories

### **4. technical_indicators.csv**
**Contains**:
- RSI, MACD, moving averages
- Bollinger Bands
- All technical indicators for all companies

### **5. investment_analysis_report_YYYYMMDD_HHMMSS.md**
**Contains**:
- Executive summary
- Company overview table
- Fundamental analysis results
- Risk analysis summary
- Technical analysis overview
- Risk warnings and disclaimers

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **PostgreSQL** database server (optional - system works without it)

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

3. **Run the setup script** (recommended):
   ```bash
   python3 scripts/setup.py
   ```
   This will:
   - Check Python version compatibility
   - Install required packages
   - Set up PostgreSQL (if available)
   - Create necessary directories
   - Configure the system

4. **Configure API keys** (optional):
   - Get a free Finnhub API key from [finnhub.io](https://finnhub.io/)
   - Get a free Alpha Vantage API key from [alphavantage.co](https://alphavantage.co/)
   - Update `scripts/config.py` with your API keys

### Running the Analysis

1. **Run the complete unified analysis**:
   ```bash
   python3 scripts/unified_pipeline.py
   ```
   This will:
   - Collect all stock and financial data
   - Perform technical analysis (RSI, MACD, moving averages)
   - Calculate risk metrics (volatility, Sharpe ratio, VaR)
   - Export ALL results to CSV files in the `results/` directory
   - Generate a comprehensive markdown report

2. **View results**:
   - Check `data/` for raw stock price data for each company
   - Check `results/` for all analysis outputs:
     - `company_summary.csv` - Company fundamentals + risk metrics
     - `stock_prices.csv` - Historical prices + returns
     - `risk_metrics.csv` - Risk analysis results
     - `technical_indicators.csv` - Technical indicators
     - `investment_analysis_report_YYYYMMDD_HHMMSS.md` - Comprehensive analysis report

## 📈 Analysis Workflow

### 1. Data Collection
- **Stock Prices**: Historical daily prices from Yahoo Finance
- **Financial Metrics**: Fundamental ratios and financial statements
- **Market Data**: Benchmark indices and market indicators

### 2. Data Processing
- **Database Storage**: All data stored in PostgreSQL for efficient querying (optional)
- **Data Cleaning**: Handle missing values and outliers
- **Feature Engineering**: Calculate technical indicators and risk metrics

### 3. Analysis Execution
- **Individual Company Analysis**: Technical, fundamental, and risk assessment
- **Comparative Analysis**: Cross-company performance comparison

### 4. Reporting
- **Text Reports**: Comprehensive analysis summaries
- **CSV Export**: Data files for further analysis

## 🗄️ Database Schema

The project uses a comprehensive PostgreSQL database with the following tables:

- **companies**: Company information and metadata
- **stock_prices**: Historical price data
- **financial_metrics**: Fundamental ratios and metrics
- **technical_indicators**: Calculated technical indicators
- **risk_metrics**: Risk analysis results
- **news_articles**: News data with sentiment scores
- **analysis_results**: Investment recommendations

## 📊 Data Analysis

### Available Data Files

The analysis generates 4 comprehensive CSV files and a detailed markdown report in the `results/` directory:

1. **company_summary.csv** - Company fundamentals and metrics
2. **stock_prices.csv** - Historical price data with returns
3. **risk_metrics.csv** - Risk analysis results
4. **technical_indicators.csv** - Technical indicators
5. **investment_analysis_report_YYYYMMDD_HHMMSS.md** - Comprehensive analysis report

### Data Usage

You can use these CSV files with any data analysis tool:
- **Excel/Google Sheets** - For basic analysis and data manipulation
- **Python** - For advanced analysis with pandas, numpy, etc.
- **R** - For statistical analysis and data processing
- **Tableau** - For interactive dashboards
- **PowerBI** - For business intelligence dashboards
- **Jupyter Notebooks** - For reproducible analysis

### Key Metrics Available

**Company Fundamentals**:
- P/E, P/B, P/S ratios
- ROE, ROA, profit margins
- Debt-to-equity ratios
- Market capitalization

**Technical Indicators**:
- RSI, MACD, moving averages
- Bollinger Bands
- Price patterns and trends

**Risk Metrics**:
- Volatility and beta
- Sharpe ratio and VaR
- Maximum drawdown
- Risk categories

## 🔧 Configuration

### Database Configuration
Update `scripts/config.py` with your database settings:
```python
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "energy_investment_db"
DB_USER = "energy_user"
DB_PASSWORD = "energy123"
```

### Analysis Parameters
Customize analysis parameters in `scripts/config.py`:
```python
# Technical Analysis Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Risk Analysis Parameters
VAR_CONFIDENCE = 0.95
SHARPE_RISK_FREE_RATE = 0.02
```

## 🔧 Technical Implementation

### Python Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance data collection
- **scipy**: Statistical computations
- **psycopg2**: PostgreSQL database connection
- **requests**: API data collection
- **textblob**: Sentiment analysis

### Key Features
- **Unified Pipeline**: All analysis centralized in `scripts/unified_pipeline.py`
- **Comprehensive Data Export**: Complete CSV dataset
- **Robust Error Handling**: Graceful handling of API failures
- **Scalable Architecture**: Easy to add new companies or metrics
- **Professional Reporting**: Investment-grade analysis outputs

### Complete Analysis Dataset
- ✅ 4 comprehensive CSV files
- ✅ All analysis results included
- ✅ All relationships preserved
- ✅ Ready for immediate analysis with any tool

## 📈 Investment Analysis Protocol Followed

### 1. **Data Collection Phase**
- Historical price data from Yahoo Finance
- Financial statements and ratios
- Market benchmark data

### 2. **Technical Analysis Phase**
- Trend analysis using moving averages
- Momentum analysis using RSI and MACD
- Volatility analysis using Bollinger Bands
- Support/resistance identification

### 3. **Fundamental Analysis Phase**
- Valuation ratio analysis
- Financial health assessment
- Growth and profitability analysis
- Dividend sustainability evaluation

### 4. **Risk Analysis Phase**
- Volatility and correlation analysis
- Beta calculation and market risk assessment
- Risk-adjusted return metrics
- Maximum drawdown analysis

### 5. **Reporting Phase**
- Comprehensive scoring system
- Investment recommendations (Buy/Hold/Sell)
- Risk warnings and disclaimers

## 📈 Next Steps

### Immediate
1. **✅ Test the unified pipeline**: `python3 scripts/unified_pipeline.py` - **COMPLETED**
2. **✅ Verify analysis data**: Check all 4 CSV files - **COMPLETED**
3. **📊 Use analysis data**: Import CSV files into your preferred analysis tool

### Future Enhancements
1. **Add more companies**: Modify `self.companies` in `unified_pipeline.py`
2. **Add new analysis**: Extend the pipeline with new methods
3. **Add real-time updates**: Schedule pipeline execution

## 🎉 Summary

The **Unified Investment Analysis Pipeline** successfully provides:

✅ **Complete CSV Export** - All results included  
✅ **Comprehensive Analysis** - Technical, fundamental, and risk analysis  
✅ **Professional Reporting** - Investment-grade analysis outputs  
✅ **Scalable Architecture** - Easy to extend and modify  

**Ready to use**: `python3 scripts/unified_pipeline.py`

---

## 📚 Additional Documentation

- `scripts/unified_pipeline.py` - Main unified pipeline
- `scripts/setup.py` - Setup and configuration script
- `scripts/config.py` - Configuration settings
- `scripts/database.py` - Database operations
- `scripts/create_tables.sql` - Database schema