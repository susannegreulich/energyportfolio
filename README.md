# Renewable Energy Investment Analysis Project

A comprehensive investment analysis system for the top 10 renewable energy companies, built with Python, SQL, and PowerBI. This project follows standard investment analysis protocols and provides detailed technical, fundamental, and risk analysis using a **Zero Duplication Unified Pipeline**.

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


The project uses a **Unified Investment Analysis Pipeline** that eliminates all code duplication and exports PowerBI data at the end using all analysis results.

### Architecture Principles
- ✅ **Zero Duplication** - Single source of truth for all calculations
- ✅ **Sequential Pipeline** - Step-by-step analysis with clear dependencies
- ✅ **Comprehensive Data Export** - PowerBI gets complete dataset with all analyses

### Pipeline Flow
```
Raw Data Collection → Technical Analysis → Risk Analysis → Portfolio Analysis → PowerBI Export
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

**Step 4: Portfolio Analysis**
- Runs portfolio optimization (Sharpe, min variance, max return)
- Calculates correlation matrix
- Uses returns from Step 1
- Stores portfolio results

**Step 5: PowerBI Export**
- Exports ALL results from Steps 1-4
- Creates 6 comprehensive CSV files
- Includes all relationships and calculated fields

## 📊 PowerBI Data Files Generated

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

### **5. portfolio_analysis.csv**
**Source**: Step 4 (Portfolio Analysis)
**Contains**:
- Portfolio weights for each optimization method
- Expected returns and volatility
- Sharpe ratios for each portfolio

### **6. correlation_matrix.csv**
**Source**: Step 4 (Portfolio Analysis)
**Contains**:
- Correlation coefficients between all companies
- Correlation categories (High/Medium/Low)

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
├── unified_pipeline.py         # Unified analysis pipeline
├── main_unified.py             # Main execution script
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
    ├── README.md              # PowerBI setup guide
    └── setup_powerbi.py       # PowerBI validation script
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


The system uses a unified pipeline with zero code duplication. All analysis is centralized in `unified_pipeline.py` and PowerBI data is exported at the end using ALL analysis results.

1. **Run the complete unified analysis** (recommended):
   ```bash
   python3 main_unified.py
   ```
   This will:
   - Collect all stock and financial data
   - Perform technical analysis (RSI, MACD, moving averages)
   - Calculate risk metrics (volatility, Sharpe ratio, VaR)
   - Run portfolio optimization
   - Export ALL results to PowerBI CSV files

2. **View results**:
   - Check `powerbi/data/` for PowerBI-ready CSV files:
     - `company_summary.csv` - Company fundamentals + risk metrics
     - `stock_prices.csv` - Historical prices + returns
     - `risk_metrics.csv` - Risk analysis results
     - `technical_indicators.csv` - Technical indicators
     - `portfolio_analysis.csv` - Portfolio optimization results
     - `correlation_matrix.csv` - Correlation analysis
   - Check `analysis/` for analysis outputs
   - Check `reports/` for generated reports
   - Check `charts/` for technical analysis charts

**Benefits of the unified pipeline**:
- ✅ **Zero Duplication** - All analysis centralized
- ✅ **Complete PowerBI Export** - All results included
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

## 📊 PowerBI Dashboard Setup

### Quick Setup (5 Minutes)

1. **Validate Your Data**:
   ```bash
   python3 powerbi/setup_powerbi.py
   ```

2. **Import Data to PowerBI**:
   - Open PowerBI Desktop
   - Click "Get Data" → "Text/CSV"
   - Import these files in order:
     - `powerbi/data/company_summary.csv`
     - `powerbi/data/risk_metrics.csv`
     - `powerbi/data/technical_indicators.csv`
     - `powerbi/data/stock_prices.csv`
     - `powerbi/data/portfolio_analysis.csv`
     - `powerbi/data/correlation_matrix.csv`

3. **Create Relationships**:
   In **Model View**, create these relationships:
   - `company_summary[ticker]` ↔ `risk_metrics[Ticker]`
   - `company_summary[ticker]` ↔ `technical_indicators[ticker]`
   - `company_summary[ticker]` ↔ `stock_prices[ticker]`

4. **Create Key Measures**:
   In **Data View**, create these DAX measures:
   ```dax
   // Portfolio Summary Measures
   Total Market Cap = SUM(company_summary[market_cap])
   Average Volatility = AVERAGE(risk_metrics[Volatility])
   Average Sharpe = AVERAGE(risk_metrics[Sharpe_Ratio])
   Average P/E = AVERAGE(company_summary[pe_ratio])

   // Risk Level Counts
   High Risk Count = COUNTROWS(FILTER(company_summary, company_summary[Risk_Category] = "High"))
   Medium Risk Count = COUNTROWS(FILTER(company_summary, company_summary[Risk_Category] = "Medium"))
   Low Risk Count = COUNTROWS(FILTER(company_summary, company_summary[Risk_Category] = "Low"))
   ```

### Essential Visualizations

#### 1. Executive Summary Page
**Company Performance Matrix (Scatter Plot)**
- X-axis: `risk_metrics[Volatility]`
- Y-axis: `risk_metrics[Sharpe_Ratio]`
- Size: `company_summary[market_cap]`
- Color: `company_summary[Risk_Category]`

**Market Cap Distribution (Treemap)**
- Group: `company_summary[company_name]`
- Values: `company_summary[market_cap]`
- Color: `company_summary[Risk_Category]`

#### 2. Technical Analysis Page
**Stock Price Chart (Line Chart)**
- X-axis: `stock_prices[date]`
- Y-axis: `stock_prices[close]`
- Legend: `stock_prices[ticker]`

**RSI Analysis (Line Chart)**
- X-axis: `technical_indicators[date]`
- Y-axis: `technical_indicators[rsi]`
- Reference lines: 30 (oversold), 70 (overbought)

#### 3. Fundamental Analysis Page
**P/E Ratios (Bar Chart)**
- Axis: `company_summary[company_name]`
- Values: `company_summary[pe_ratio]`

**ROE Comparison (Bar Chart)**
- Axis: `company_summary[company_name]`
- Values: `company_summary[roe]`

#### 4. Risk Analysis Page
**Risk-Return Scatter Plot**
- X-axis: `risk_metrics[Volatility]`
- Y-axis: `risk_metrics[Annual_Return]`
- Size: `company_summary[market_cap]`
- Color: `risk_metrics[Risk_Level]`

### Design Guidelines

**Color Scheme**:
- **Risk Levels**: 
  - Low/Medium: Green (#00B050)
  - High: Orange (#FF6600)
  - Very High: Red (#C00000)
- **Background**: Light gray (#F8F9FA)
- **Text**: Dark gray (#212529)

**Layout**:
- Use **12-column grid**
- **Consistent spacing** (10px margins)
- **Card design** with subtle shadows
- **Responsive sizing** for different screen sizes

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
# Technical Analysis Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Risk Analysis Parameters
VAR_CONFIDENCE = 0.95
SHARPE_RISK_FREE_RATE = 0.02

# Portfolio Optimization Parameters
OPTIMIZATION_METHODS = ['sharpe', 'min_variance', 'max_return']
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

### 5. **Portfolio Optimization Phase**
- Multi-objective optimization
- Risk-return trade-off analysis
- Diversification assessment
- Backtesting and validation

### 6. **Recommendation Phase**
- Comprehensive scoring system
- Investment recommendations (Buy/Hold/Sell)
- Risk warnings and disclaimers
- Portfolio allocation suggestions

## 🔧 Technical Implementation

### Python Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **yfinance**: Yahoo Finance data collection
- **scikit-learn**: Machine learning and optimization
- **matplotlib/seaborn**: Data visualization
- **psycopg2**: PostgreSQL database connection
- **requests**: API data collection
- **textblob**: Sentiment analysis

### Key Features
- **Zero Duplication**: All analysis centralized in unified pipeline
- **Comprehensive Data Export**: Complete PowerBI dataset
- **Robust Error Handling**: Graceful handling of API failures
- **Scalable Architecture**: Easy to add new companies or metrics
- **Professional Reporting**: Investment-grade analysis outputs

### Complete PowerBI Dataset
- ✅ 6 comprehensive CSV files
- ✅ All analysis results included
- ✅ All relationships preserved
- ✅ Ready for immediate PowerBI import

### Maintainability Improved
- ✅ Single source of truth
- ✅ Clear data flow
- ✅ Easy to modify and extend
- ✅ Comprehensive logging

## 📈 Next Steps

### Immediate
1. **✅ Test the unified pipeline**: `python3 main_unified.py` - **COMPLETED**
2. **✅ Verify PowerBI data**: Check all 6 CSV files - **COMPLETED**
3. **📊 Import to PowerBI**: Use the data for dashboard creation

### Future Enhancements
1. **Add more companies**: Modify `self.companies` in `unified_pipeline.py`
2. **Add new analysis**: Extend the pipeline with new methods
3. **Add database storage**: Optionally save results to database
4. **Add real-time updates**: Schedule pipeline execution

### PowerBI Setup
1. Open PowerBI Desktop
2. Import CSV files from `powerbi/data/` directory
3. Follow the PowerBI setup guide in `powerbi/` directory
4. Create your interactive dashboard!

## 🎉 Summary

The **Unified Investment Analysis Pipeline** successfully achieves:

✅ **Complete PowerBI Export** - All results included  
✅ **Better Maintainability** - Single source of truth  
✅ **Improved Performance** - No redundant calculations  
✅ **Enhanced Reliability** - Consistent results  

**Ready to use**: `python3 main_unified.py`

---

## 📚 Additional Documentation

- `powerbi/README.md` - PowerBI setup guide
- `powerbi/PowerBI_Quick_Start.md` - 5-minute setup
- `powerbi/PowerBI_Dashboard_Guide.md` - Comprehensive guide
- `powerbi/sample_dashboard_structure.md` - Detailed structure
- `unified_pipeline.py` - Main unified pipeline (928 lines)
- `main_unified.py` - Simple execution script