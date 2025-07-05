# Script Structure Documentation

## Overview

The project has been restructured to separate different analysis steps into individual scripts, making the codebase more modular and maintainable. The `main.py` script now serves as a simple orchestrator that calls each step in the correct order.

## Script Organization

### 1. `main.py` (Orchestrator)
- **Purpose**: Coordinates the entire analysis pipeline
- **Size**: ~80 lines (down from 475 lines)
- **Function**: Calls individual scripts in sequence and handles errors
- **Usage**: `python main.py`

### 2. `fetch_data.py` (Data Setup & Collection)
- **Purpose**: Handles database initialization, data loading, and data collection
- **Size**: ~660 lines
- **Function**: 
  - Creates output directories
  - Initializes database connection
  - Runs unified data pipeline
  - Handles data fetching and storage
  - Exports data to PowerBI CSV files
  - Loads data into database
- **Usage**: `python fetch_data.py`

### 3. `technical_charts.py` (Technical Analysis)
- **Purpose**: Generates technical analysis charts for individual companies
- **Size**: ~120 lines
- **Function**:
  - Creates technical indicators (SMA, RSI, MACD, Bollinger Bands)
  - Generates multi-panel charts for each company
  - Saves charts to `charts/` directory
- **Usage**: `python technical_charts.py`

### 4. `risk_analysis_charts.py` (Risk Analysis)
- **Purpose**: Generates risk analysis charts for all companies
- **Size**: ~110 lines
- **Function**:
  - Calculates risk metrics (volatility, Sharpe ratio, drawdown, VaR)
  - Creates comparison charts across companies
  - Generates correlation heatmap
- **Usage**: `python risk_analysis_charts.py`

### 5. `portfolio_optimization.py` (Portfolio Analysis)
- **Purpose**: Performs portfolio optimization analysis
- **Size**: ~150 lines
- **Function**:
  - Implements multiple optimization methods (Sharpe, Min Variance, Max Return)
  - Creates portfolio visualization charts
  - Performs backtesting
- **Usage**: `python portfolio_optimization.py`

### 6. `report_generator.py` (Report Generation)
- **Purpose**: Generates comprehensive analysis reports
- **Size**: ~140 lines
- **Function**:
  - Creates detailed text reports
  - Includes individual company analysis
  - Summarizes portfolio optimization results
  - Provides investment recommendations
- **Usage**: `python report_generator.py`

## Execution Flow

```
main.py
├── fetch_data.py
├── technical_charts.py
├── risk_analysis_charts.py
├── portfolio_optimization.py
└── report_generator.py
```

## Benefits of This Structure

1. **Modularity**: Each script has a single responsibility
2. **Maintainability**: Easier to debug and modify individual components
3. **Reusability**: Scripts can be run independently
4. **Scalability**: Easy to add new analysis steps
5. **Testing**: Individual components can be tested separately
6. **Parallel Development**: Different team members can work on different scripts

## Running Individual Scripts

You can run any script independently:

```bash
# Run only data setup
python fetch_data.py

# Run only technical analysis
python technical_charts.py

# Run only risk analysis
python risk_analysis_charts.py

# Run only portfolio optimization
python portfolio_optimization.py

# Run only report generation
python report_generator.py
```

## Dependencies

Each script imports the necessary modules:
- `config.py` - Configuration settings
- `database.py` - Database management
- `fetch_data.py` - Data fetching utilities
- `investment_analysis.py` - Core analysis functions

## Output Structure

```
project/
├── main.py                    # Orchestrator
├── fetch_data.py    # Data setup & collection script
├── technical_charts.py       # Technical analysis script
├── risk_analysis_charts.py   # Risk analysis script
├── portfolio_optimization.py # Portfolio optimization script
├── report_generator.py       # Report generation script
├── charts/                   # Generated charts
├── reports/                  # Generated reports
├── data/                     # Data files
└── powerbi/                  # PowerBI data
```

## Error Handling

- Each script has its own error handling
- The orchestrator (`main.py`) captures and reports errors from each step
- Failed steps are clearly identified with error messages
- The pipeline stops if any step fails

## Future Enhancements

This structure makes it easy to add new analysis steps:
1. Create a new script (e.g., `sentiment_analysis.py`)
2. Add it to the steps list in `main.py`
3. The new step will be automatically included in the pipeline 