# Renewable Energy Investment Analysis Project

A comprehensive investment analysis system for the top 10 renewable energy companies, built with Python and SQL. This project follows standard investment analysis protocols and provides detailed technical, fundamental, and risk analysis using a **Zero Duplication Unified Pipeline**.

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


The project uses a **Unified Investment Analysis Pipeline** that eliminates all code duplication and exports comprehensive CSV data at the end using all analysis results.

### Architecture Principles
- ✅ **Zero Duplication** - Single source of truth for all calculations
- ✅ **Sequential Pipeline** - Step-by-step analysis with clear dependencies
- ✅ **Comprehensive Data Export** - CSV files with complete dataset and all analyses

### Pipeline Flow
```
Raw Data Collection → Technical Analysis → Risk Analysis → CSV Export
```

**Step 1: Data Collection**
- Fetches stock prices, financial statements, company info
- Calculates fundamental metrics (P/E, ROE, etc.)
- Stores all raw data in memory

**Step 2: Technical Analysis**
- Calculates RSI, MACD, moving averages, Bollinger Bands
- Uses stock price data from Step 1
- Stores technical indicators

**Step 3: Risk Analysis**
- Calculates volatility, Sharpe ratio, VaR, maximum drawdown
- Uses returns calculated from Step 1
- Stores risk metrics

**Step 4: CSV Export**
- Exports ALL results from Steps 1-3
- Creates 4 comprehensive CSV files
- Includes all relationships and calculated fields

## 📊 Analysis Data Files Generated

### **1. company_summary.csv**
**Source**: Step 1 (Data Collection)
**Contains**:
- Company fundamentals (P/E, ROE, debt-to-equity)
- Market data (price, market cap)
- Risk metrics (volatility, Sharpe ratio)
- Calculated fields (market cap billions, risk categories)

### **2. stock_prices.csv**
**Source**: Step 1 (Data Collection)
**Contains**:
- Historical OHLCV data
- Calculated returns (daily, cumulative)
- Time dimensions (year, month, quarter)

### **3. risk_metrics.csv**
**Source**: Step 3 (Risk Analysis)
**Contains**:
- Volatility, annual return, Sharpe ratio
- Maximum drawdown, VaR, CVaR
- Risk levels and categories

### **4. technical_indicators.csv**
**Source**: Step 2 (Technical Analysis)
**Contains**:
- RSI, MACD, moving averages
- Bollinger Bands
- All technical indicators for all companies



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



## 🏗️ Project Structure

```
energy_investment_project/
├── scripts/
│   ├── config.py              # Configuration settings
│   ├── database.py            # Database connection and operations
│   ├── unified_pipeline.py    # Unified analysis pipeline
│   ├── setup.py               # Setup and configuration script
│   └── create_tables.sql      # Database schema
├── requirements.txt           # Python dependencies
├── README.md                 # This file

├── data/
│   └── raw/                  # Raw data storage
├── results/                  # All analysis outputs (CSV files and reports)
└── charts/                   # Generated charts
```

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

3. **Configure the database** (optional):
   - Update `scripts/config.py` with your PostgreSQL credentials
   - Ensure PostgreSQL is running
   - System will work without database using direct data collection

4. **Set up API keys** (optional):
   - Get a free Finnhub API key from [finnhub.io](https://finnhub.io/)
   - Update `scripts/config.py` with your API key

### Running the Analysis


The system uses a unified pipeline with zero code duplication. All analysis is centralized in `scripts/unified_pipeline.py` and CSV data is exported at the end using ALL analysis results.

1. **Run the complete unified analysis** (recommended):
   ```bash
   python3 scripts/unified_pipeline.py
   ```
   This will:
   - Collect all stock and financial data
   - Perform technical analysis (RSI, MACD, moving averages)
   - Calculate risk metrics (volatility, Sharpe ratio, VaR)
   - Export ALL results to CSV files in the analysis/ directory

2. **View results**:
   - Check `results/` for all analysis outputs:
     - `company_summary.csv` - Company fundamentals + risk metrics
     - `stock_prices.csv` - Historical prices + returns
     - `risk_metrics.csv` - Risk analysis results
     - `technical_indicators.csv` - Technical indicators
     - `investment_analysis_report_YYYYMMDD_HHMMSS.md` - Comprehensive analysis report
   - Check `charts/` for technical analysis charts

**Benefits of the unified pipeline**:
- ✅ **Zero Duplication** - All analysis centralized
- ✅ **Complete CSV Export** - All results included
- ✅ **Better Performance** - No redundant calculations
- ✅ **Enhanced Reliability** - Single source of truth

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
- **Comparative Analysis**: Cross-company performance comparison

### 4. Reporting
- **Text Reports**: Comprehensive analysis summaries
- **Visual Charts**: Technical analysis and risk visualization
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

### Data Usage

You can use these CSV files with any data analysis tool:
- **Excel/Google Sheets** - For basic analysis and visualization
- **Python** - For advanced analysis with pandas, matplotlib, etc.
- **R** - For statistical analysis and visualization
- **Tableau** - For interactive dashboards
- **PowerBI** - For business intelligence dashboards (optional)
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
DB_USER = "your_username"
DB_PASSWORD = "your_password"
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

## 📊 Investment Analysis Protocol Followed

### 1. **Data Collection Phase**
- Historical price data from multiple sources
- Financial statements and ratios
- News sentiment analysis
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



### 6. **Recommendation Phase**
- Comprehensive scoring system
- Investment recommendations (Buy/Hold/Sell)
- Risk warnings and disclaimers


## 🔧 Technical Implementation

### Python Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance data collection
- **scipy**: Statistical computations
- **matplotlib/seaborn**: Data visualization
- **psycopg2**: PostgreSQL database connection
- **requests**: API data collection
- **textblob**: Sentiment analysis

### Key Features
- **Zero Duplication**: All analysis centralized in scripts/unified_pipeline.py
- **Comprehensive Data Export**: Complete CSV dataset
- **Robust Error Handling**: Graceful handling of API failures
- **Scalable Architecture**: Easy to add new companies or metrics
- **Professional Reporting**: Investment-grade analysis outputs

### Complete Analysis Dataset
- ✅ 4 comprehensive CSV files
- ✅ All analysis results included
- ✅ All relationships preserved
- ✅ Ready for immediate analysis with any tool

### Maintainability Improved
- ✅ Single source of truth
- ✅ Clear data flow
- ✅ Easy to modify and extend
- ✅ Comprehensive logging

## 📈 Next Steps

### Immediate
1. **✅ Test the unified pipeline**: `python3 main.py` - **COMPLETED**
2. **✅ Verify analysis data**: Check all 4 CSV files - **COMPLETED**
3. **📊 Use analysis data**: Import CSV files into your preferred analysis tool

### Future Enhancements
1. **Add more companies**: Modify `self.companies` in `unified_pipeline.py`
2. **Add new analysis**: Extend the pipeline with new methods
3. **Add database storage**: Optionally save results to database
4. **Add real-time updates**: Schedule pipeline execution

## 🎉 Summary

The **Unified Investment Analysis Pipeline** successfully achieves:

✅ **Complete CSV Export** - All results included  
✅ **Better Maintainability** - Single source of truth  
✅ **Improved Performance** - No redundant calculations  
✅ **Enhanced Reliability** - Consistent results  

**Ready to use**: `python3 main.py`

---

## 📚 Additional Documentation

- `unified_pipeline.py` - Main unified pipeline
- `main.py` - Simple execution script
- `config.py` - Configuration settings
- `database.py` - Database operations (optional)