# PowerBI Dashboard Setup - Energy Investment Analysis

## 🎯 Overview

This directory contains everything you need to create a comprehensive PowerBI dashboard for your renewable energy investment analysis. You have all the data ready and now need to build the visualizations in PowerBI Desktop.

## 📁 Files in This Directory

### 📊 Data Files (Ready for Import)
- **`data/company_summary.csv`** - Company fundamentals and metrics (9 companies)
- **`data/stock_prices.csv`** - Historical price data (15,904 records)
- **`data/risk_metrics.csv`** - Risk analysis results (9 companies)
- **`data/technical_indicators.csv`** - Technical indicators (15,904 records)

### 📚 Documentation
- **`PowerBI_Dashboard_Guide.md`** - Comprehensive guide with detailed instructions
- **`PowerBI_Quick_Start.md`** - Quick setup guide (5-minute start)
- **`sample_dashboard_structure.md`** - Detailed dashboard structure template
- **`setup_powerbi.py`** - Python script to validate data files

## 🚀 Quick Start (5 Minutes)

### Step 1: Validate Your Data
```bash
python3 powerbi/setup_powerbi.py
```

### Step 2: Open PowerBI Desktop
1. Download and install PowerBI Desktop if you haven't already
2. Open PowerBI Desktop

### Step 3: Import Data
1. Click **"Get Data"** → **"Text/CSV"**
2. Import these files in order:
   - `powerbi/data/company_summary.csv`
   - `powerbi/data/risk_metrics.csv`
   - `powerbi/data/technical_indicators.csv`
   - `powerbi/data/stock_prices.csv`

### Step 4: Create Relationships
In **Model View**, create these relationships:
- `company_summary[ticker]` ↔ `risk_metrics[Ticker]`
- `company_summary[ticker]` ↔ `technical_indicators[ticker]`
- `company_summary[ticker]` ↔ `stock_prices[ticker]`

### Step 5: Create Key Measures
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

## 📊 Essential Visualizations to Start With

### 1. Executive Summary Page
**Company Performance Matrix (Scatter Plot)**
- X-axis: `risk_metrics[Volatility]`
- Y-axis: `risk_metrics[Sharpe_Ratio]`
- Size: `company_summary[market_cap]`
- Color: `company_summary[Risk_Category]`

**Market Cap Distribution (Treemap)**
- Group: `company_summary[company_name]`
- Values: `company_summary[market_cap]`
- Color: `company_summary[Risk_Category]`

### 2. Technical Analysis Page
**Stock Price Chart (Line Chart)**
- X-axis: `stock_prices[date]`
- Y-axis: `stock_prices[close]`
- Legend: `stock_prices[ticker]`

**RSI Analysis (Line Chart)**
- X-axis: `technical_indicators[date]`
- Y-axis: `technical_indicators[rsi]`
- Reference lines: 30 (oversold), 70 (overbought)

### 3. Fundamental Analysis Page
**P/E Ratios (Bar Chart)**
- Axis: `company_summary[company_name]`
- Values: `company_summary[pe_ratio]`

**ROE Comparison (Bar Chart)**
- Axis: `company_summary[company_name]`
- Values: `company_summary[roe]`

### 4. Risk Analysis Page
**Risk-Return Scatter Plot**
- X-axis: `risk_metrics[Volatility]`
- Y-axis: `risk_metrics[Annual_Return]`
- Size: `company_summary[market_cap]`
- Color: `risk_metrics[Risk_Level]`

## 🎨 Design Guidelines

### Color Scheme
- **Risk Levels**: 
  - Low/Medium: Green (#00B050)
  - High: Orange (#FF6600)
  - Very High: Red (#C00000)
- **Background**: Light gray (#F8F9FA)
- **Text**: Dark gray (#212529)

### Layout
- Use **12-column grid**
- **Consistent spacing** (10px margins)
- **Card design** with subtle shadows
- **Responsive sizing** for different screen sizes

## 🔧 Advanced Features

### Filters and Slicers
1. **Date Range Slicer**: For time-based analysis
2. **Company Multi-Select**: For comparing specific companies
3. **Risk Level Filter**: For risk-based filtering

### Drill-Down Capabilities
1. **Company Drill-Down**: Click company to see detailed analysis
2. **Time Drill-Down**: Drill from year to quarter to month
3. **Metric Drill-Down**: Drill from summary to detailed metrics

### Bookmarks
Create bookmarks for:
1. **Default View**: Standard dashboard view
2. **Risk Focus**: Highlight risk analysis
3. **Technical Focus**: Highlight technical analysis
4. **Fundamental Focus**: Highlight fundamental analysis

## 📱 Interactive Features

### Custom Tooltips
Create rich tooltips showing:
- Company name and ticker
- Key metrics summary
- Performance indicators
- Risk warnings

### Conditional Formatting
- Color-code cells based on performance
- Icon indicators for trends
- Data bars for comparisons

### Dynamic Titles
- Update titles based on selected filters
- Show current date/time
- Display selected company names

## 🚀 Performance Optimization

### Data Refresh
- Set up automatic refresh schedule
- Use incremental refresh for large datasets
- Optimize query performance

### File Size
- Compress images and icons
- Use efficient data types
- Remove unused columns

## 📊 Sample Queries

### Top Performers
```dax
Top Performers = 
TOPN(5, company_summary, company_summary[sharpe_ratio], DESC)
```

### High Risk Companies
```dax
High Risk Companies = 
FILTER(company_summary, company_summary[Risk_Category] = "High")
```

### Undervalued Companies
```dax
Undervalued = 
FILTER(company_summary, 
    company_summary[pe_ratio] < 20 && 
    company_summary[roe] > 0.1)
```

## 🎯 Success Checklist

- [ ] All 4 CSV files imported
- [ ] Relationships created between tables
- [ ] Key measures created
- [ ] Executive Summary page built
- [ ] Technical Analysis page built
- [ ] Fundamental Analysis page built
- [ ] Risk Analysis page built
- [ ] Filters and slicers working
- [ ] Drill-down capabilities tested
- [ ] Performance optimized
- [ ] Dashboard shared with team

## 📚 Detailed Documentation

### For Complete Instructions
- **`PowerBI_Dashboard_Guide.md`** - Comprehensive guide with all details
- **`sample_dashboard_structure.md`** - Detailed structure for each page

### For Quick Reference
- **`PowerBI_Quick_Start.md`** - Quick setup and essential visualizations

## 🔄 Data Updates

### Automatic Updates
Your Python analysis pipeline automatically updates the CSV files when you run:
```bash
python3 fetch_data.py
```

### Manual Updates
1. Run the data collection script
2. Refresh the PowerBI dashboard
3. Verify all data is current

## 🆘 Troubleshooting

### Common Issues
1. **Data Loading Errors**: Check CSV file format and encoding
2. **Relationship Errors**: Verify foreign keys match exactly
3. **Performance Issues**: Optimize data model and queries
4. **Visualization Errors**: Check data types and null values

### Support Resources
- PowerBI documentation: https://docs.microsoft.com/power-bi/
- Community forums: https://community.powerbi.com/
- Microsoft support: https://support.microsoft.com/

## 📈 Next Steps

1. **Start with Executive Summary**: Build the overview page first
2. **Add Technical Analysis**: Create price charts and indicators
3. **Include Fundamental Analysis**: Add financial metrics
4. **Complete Risk Analysis**: Add risk assessment visualizations
5. **Test Interactions**: Verify all filters and drill-downs work
6. **Optimize Performance**: Ensure fast loading times
7. **Get Feedback**: Share with stakeholders for input
8. **Iterate**: Make improvements based on feedback

## 🎉 You're Ready!

You have all the data and documentation needed to create an amazing PowerBI dashboard for your renewable energy investment analysis. The data is comprehensive, the structure is well-defined, and the visualizations will provide valuable insights for investment decisions.

**Start with the Quick Start guide and build your dashboard step by step!**

---

*For questions or issues, refer to the detailed documentation files in this directory.* 