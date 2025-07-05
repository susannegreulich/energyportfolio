-- Renewable Energy Investment Analysis Database Schema
-- PostgreSQL Database Setup

-- Create database (run this separately if needed)
-- CREATE DATABASE energy_investment_db;

-- Connect to the database
-- \c energy_investment_db;

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    company_id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) UNIQUE NOT NULL,
    company_name VARCHAR(100) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(50),
    country VARCHAR(50),
    market_cap DECIMAL(20,2),
    employee_count INTEGER,
    website VARCHAR(200),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stock prices table
CREATE TABLE IF NOT EXISTS stock_prices (
    price_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    date DATE NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    adjusted_close DECIMAL(10,4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- Financial metrics table
CREATE TABLE IF NOT EXISTS financial_metrics (
    metric_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    date DATE NOT NULL,
    pe_ratio DECIMAL(10,4),
    pb_ratio DECIMAL(10,4),
    ps_ratio DECIMAL(10,4),
    debt_to_equity DECIMAL(10,4),
    current_ratio DECIMAL(10,4),
    quick_ratio DECIMAL(10,4),
    roe DECIMAL(10,4),
    roa DECIMAL(10,4),
    profit_margin DECIMAL(10,4),
    revenue_growth DECIMAL(10,4),
    earnings_growth DECIMAL(10,4),
    dividend_yield DECIMAL(10,4),
    payout_ratio DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- News articles table
CREATE TABLE IF NOT EXISTS news_articles (
    article_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    headline TEXT NOT NULL,
    summary TEXT,
    url VARCHAR(500),
    source VARCHAR(100),
    published_date TIMESTAMP,
    sentiment_score DECIMAL(3,2),
    relevance_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS technical_indicators (
    indicator_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    date DATE NOT NULL,
    sma_20 DECIMAL(10,4),
    sma_50 DECIMAL(10,4),
    sma_200 DECIMAL(10,4),
    rsi DECIMAL(5,2),
    macd DECIMAL(10,4),
    macd_signal DECIMAL(10,4),
    macd_histogram DECIMAL(10,4),
    bollinger_upper DECIMAL(10,4),
    bollinger_lower DECIMAL(10,4),
    bollinger_middle DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- Risk metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    risk_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    date DATE NOT NULL,
    volatility DECIMAL(10,6),
    beta DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    var_95 DECIMAL(10,6),
    cvar_95 DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- Portfolio performance table
CREATE TABLE IF NOT EXISTS portfolio_performance (
    performance_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_value DECIMAL(15,2),
    daily_return DECIMAL(10,6),
    cumulative_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    benchmark_return DECIMAL(10,6),
    excess_return DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Portfolio holdings table
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    holding_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    date DATE NOT NULL,
    shares_owned DECIMAL(15,4),
    share_price DECIMAL(10,4),
    market_value DECIMAL(15,2),
    weight DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date)
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    result_id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    analysis_date DATE NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    score DECIMAL(5,2),
    rank INTEGER,
    recommendation VARCHAR(20),
    confidence_level DECIMAL(3,2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance (after all tables are created)
-- These will be created separately to avoid dependency issues

-- Create views for common queries
CREATE OR REPLACE VIEW company_summary AS
SELECT 
    c.company_id,
    c.ticker,
    c.company_name,
    c.sector,
    c.industry,
    c.market_cap,
    sp.close_price as current_price,
    sp.date as last_price_date,
    fm.pe_ratio,
    fm.pb_ratio,
    fm.roe,
    fm.debt_to_equity,
    rm.volatility,
    rm.beta,
    rm.sharpe_ratio
FROM companies c
LEFT JOIN (
    SELECT DISTINCT ON (company_id) company_id, close_price, date
    FROM stock_prices
    ORDER BY company_id, date DESC
) sp ON c.company_id = sp.company_id
LEFT JOIN (
    SELECT DISTINCT ON (company_id) company_id, pe_ratio, pb_ratio, roe, debt_to_equity
    FROM financial_metrics
    ORDER BY company_id, date DESC
) fm ON c.company_id = fm.company_id
LEFT JOIN (
    SELECT DISTINCT ON (company_id) company_id, volatility, beta, sharpe_ratio
    FROM risk_metrics
    ORDER BY company_id, date DESC
) rm ON c.company_id = rm.company_id;

-- Create view for portfolio analysis
CREATE OR REPLACE VIEW portfolio_analysis AS
SELECT 
    ph.date,
    c.ticker,
    c.company_name,
    ph.shares_owned,
    ph.share_price,
    ph.market_value,
    ph.weight,
    pp.total_value,
    pp.daily_return,
    pp.cumulative_return
FROM portfolio_holdings ph
JOIN companies c ON ph.company_id = c.company_id
JOIN portfolio_performance pp ON ph.date = pp.date
ORDER BY ph.date DESC, ph.weight DESC;
