#!/usr/bin/env python3
"""
PowerBI Setup and Validation Script
Validates and prepares CSV files for PowerBI import
"""

import pandas as pd
import os
import sys
from pathlib import Path

def validate_csv_file(file_path, expected_columns=None):
    """Validate a CSV file for PowerBI import"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ {os.path.basename(file_path)}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
            else:
                print(f"   ✅ All expected columns present")
        
        # Check for data quality issues
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"   ⚠️  Null values found: {null_counts.sum():,} total")
        
        return True, df
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False, None

def create_powerbi_summary():
    """Create a summary of all PowerBI data files"""
    powerbi_dir = Path("powerbi/data")
    
    if not powerbi_dir.exists():
        print("❌ PowerBI data directory not found!")
        return False
    
    csv_files = list(powerbi_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in powerbi/data/")
        return False
    
    print("=" * 60)
    print("POWERBI DATA VALIDATION REPORT")
    print("=" * 60)
    
    # Expected columns for each file
    expected_columns = {
        "company_summary.csv": ["ticker", "company_name", "market_cap", "current_price", "pe_ratio", "roe", "volatility", "risk_category"],
        "stock_prices.csv": ["ticker", "date", "open", "high", "low", "close", "volume"],
        "risk_metrics.csv": ["ticker", "company_name", "volatility", "annual_return", "sharpe_ratio", "max_drawdown", "var_95", "beta", "risk_level"],
        "technical_indicators.csv": ["ticker", "date", "rsi", "macd", "macd_signal", "sma_20", "sma_50", "sma_200"]
    }
    
    all_valid = True
    dataframes = {}
    
    for csv_file in csv_files:
        filename = csv_file.name
        expected_cols = expected_columns.get(filename, None)
        
        is_valid, df = validate_csv_file(csv_file, expected_cols)
        if is_valid:
            dataframes[filename] = df
        else:
            all_valid = False
        
        print()
    
    if all_valid:
        print("✅ All CSV files are valid and ready for PowerBI import!")
        create_import_guide(dataframes)
    else:
        print("❌ Some files have issues. Please fix before importing to PowerBI.")
    
    return all_valid

def create_import_guide(dataframes):
    """Create a guide for importing data into PowerBI"""
    print("\n" + "=" * 60)
    print("POWERBI IMPORT GUIDE")
    print("=" * 60)
    
    print("\n📊 Step 1: Import Data Files")
    print("1. Open PowerBI Desktop")
    print("2. Click 'Get Data' → 'Text/CSV'")
    print("3. Import these files in order:")
    
    for i, filename in enumerate(dataframes.keys(), 1):
        print(f"   {i}. {filename}")
    
    print("\n📊 Step 2: Data Modeling")
    print("Create these relationships in PowerBI:")
    print("• company_summary[ticker] ↔ stock_prices[ticker]")
    print("• company_summary[ticker] ↔ risk_metrics[ticker]")
    print("• company_summary[ticker] ↔ technical_indicators[ticker]")
    
    print("\n📊 Step 3: Key Measures to Create")
    print("Use these DAX formulas:")
    print("""
// Average Volatility
Avg Volatility = AVERAGE(risk_metrics[Volatility])

// Average Sharpe Ratio  
Avg Sharpe = AVERAGE(risk_metrics[Sharpe_Ratio])

// Total Market Cap
Total Market Cap = SUM(company_summary[market_cap])

// Risk Level Count
High Risk Count = COUNTROWS(FILTER(company_summary, company_summary[Risk_Category] = "High"))
Medium Risk Count = COUNTROWS(FILTER(company_summary, company_summary[Risk_Category] = "Medium"))
Low Risk Count = COUNTROWS(FILTER(company_summary, company_summary[Risk_Category] = "Low"))
    """)
    
    print("\n📊 Step 4: Recommended Visualizations")
    print("Start with these key charts:")
    print("1. Scatter Plot: Volatility vs Sharpe Ratio (size by Market Cap)")
    print("2. Bar Chart: P/E Ratios by Company")
    print("3. Line Chart: Stock Prices over Time")
    print("4. Treemap: Market Cap Distribution")
    print("5. Table: Company Summary with Key Metrics")

def create_sample_queries():
    """Create sample PowerBI queries for common analysis"""
    print("\n" + "=" * 60)
    print("SAMPLE POWERBI QUERIES")
    print("=" * 60)
    
    queries = {
        "Top Performers": """
// Top 5 companies by Sharpe Ratio
Top Performers = 
TOPN(5, company_summary, company_summary[sharpe_ratio], DESC)
""",
        
        "Risk Analysis": """
// Companies with highest volatility
High Risk Companies = 
FILTER(company_summary, company_summary[volatility] > 0.4)
""",
        
        "Valuation Analysis": """
// Undervalued companies (low P/E, high ROE)
Undervalued = 
FILTER(company_summary, 
    company_summary[pe_ratio] < 20 && 
    company_summary[roe] > 0.1)
""",
        
        "Portfolio Summary": """
// Portfolio summary metrics
Portfolio Summary = 
SUMMARIZE(company_summary,
    "Total Market Cap", SUM(company_summary[market_cap]),
    "Avg P/E", AVERAGE(company_summary[pe_ratio]),
    "Avg ROE", AVERAGE(company_summary[roe]),
    "Avg Volatility", AVERAGE(company_summary[volatility])
)
"""
    }
    
    for query_name, query in queries.items():
        print(f"\n📊 {query_name}:")
        print(query)

def main():
    """Main function"""
    print("🔍 PowerBI Setup and Validation")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("powerbi").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Validate all CSV files
    success = create_powerbi_summary()
    
    if success:
        create_sample_queries()
        
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Open PowerBI Desktop")
        print("2. Follow the import guide above")
        print("3. Create visualizations using the dashboard guide")
        print("4. Test all interactions and filters")
        print("\n📚 For detailed instructions, see: powerbi/PowerBI_Dashboard_Guide.md")
    
    return success

if __name__ == "__main__":
    main() 