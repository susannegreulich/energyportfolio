"""
Database connection and data loading module for Renewable Energy Investment Analysis
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
from datetime import date as dt_date
import os
from config import *

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for handling all database operations"""
    
    def __init__(self):
        """Initialize database connection"""
        self.connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        self.engine = None
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            self.connection = self.engine.connect()
            logger.info("Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            logger.warning("Database connection failed. The application will continue without database functionality.")
            logger.warning("To fix this issue:")
            logger.warning("1. Ensure PostgreSQL is running")
            logger.warning("2. Update config.py with correct database credentials")
            logger.warning("3. Create the database: createdb energy_investment_db")
            logger.warning("4. Or run setup.py to configure the database automatically")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return results"""
        try:
            if params:
                result = self.connection.execute(text(query), params)
            else:
                result = self.connection.execute(text(query))
            
            if result.returns_rows:
                return result.fetchall()
            else:
                self.connection.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.connection.rollback()
            raise
    
    def load_companies(self, companies_dict):
        """Load company information into database"""
        query = """
        INSERT INTO companies (ticker, company_name, sector, industry, country, market_cap, description)
        VALUES (:ticker, :company_name, :sector, :industry, :country, :market_cap, :description)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            country = EXCLUDED.country,
            market_cap = EXCLUDED.market_cap,
            description = EXCLUDED.description,
            updated_at = CURRENT_TIMESTAMP
        """
        
        for name, ticker in companies_dict.items():
            # Determine sector and country based on ticker
            if any(ext in ticker for ext in ['.CO']):
                country = 'Denmark'
                sector = 'Renewable Energy'
            elif any(ext in ticker for ext in ['.MC']):
                country = 'Spain'
                sector = 'Renewable Energy'
            elif any(ext in ticker for ext in ['.LS']):
                country = 'Portugal'
                sector = 'Renewable Energy'
            elif any(ext in ticker for ext in ['.MI']):
                country = 'Italy'
                sector = 'Renewable Energy'
            else:
                country = 'United States'
                sector = 'Renewable Energy'
            
            params = {
                'ticker': ticker,
                'company_name': name,
                'sector': sector,
                'industry': 'Renewable Energy',
                'country': country,
                'market_cap': None,
                'description': f'{name} - Renewable Energy Company'
            }
            self.execute_query(query, params)
        
        logger.info(f"Loaded {len(companies_dict)} companies into database")
    
    def _convert_numpy_types(self, value):
        """Convert numpy types to native Python types"""
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        elif hasattr(value, 'tolist'):  # numpy array
            return value.tolist()
        elif pd.isna(value):  # pandas NaN
            return None
        else:
            return value
    
    def _convert_decimal_types(self, value):
        """Convert Decimal types to float for analysis"""
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
        elif hasattr(value, 'item'):  # numpy scalar
            return value.item()
        elif pd.isna(value):  # pandas NaN
            return None
        else:
            return value
    
    def load_stock_prices(self, ticker, price_data):
        """Load stock price data into database"""
        # Get company_id
        company_query = "SELECT company_id FROM companies WHERE ticker = :ticker"
        result = self.execute_query(company_query, {'ticker': ticker})
        if not result:
            logger.warning(f"Company {ticker} not found in database")
            return
        
        company_id = result[0][0]
        
        # Prepare data for insertion
        query = """
        INSERT INTO stock_prices (company_id, date, open_price, high_price, low_price, close_price, adjusted_close, volume)
        VALUES (:company_id, :date, :open_price, :high_price, :low_price, :close_price, :adjusted_close, :volume)
        ON CONFLICT (company_id, date) DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume
        """
        
        for date, row in price_data.iterrows():
            adj_close = row.get('Adj Close')
            if adj_close is None:
                adj_close = row.get('Adj. Close')
            if adj_close is None:
                adj_close = row.get('Close')
            
            # Convert numpy types to native Python types
            params = {
                'company_id': company_id,
                'date': date if isinstance(date, dt_date) else date.date(),
                'open_price': self._convert_numpy_types(row['Open']),
                'high_price': self._convert_numpy_types(row['High']),
                'low_price': self._convert_numpy_types(row['Low']),
                'close_price': self._convert_numpy_types(row['Close']),
                'adjusted_close': self._convert_numpy_types(adj_close),
                'volume': self._convert_numpy_types(row['Volume'])
            }
            self.execute_query(query, params)
        
        logger.info(f"Loaded {len(price_data)} price records for {ticker}")
    
    def load_financial_metrics(self, ticker, metrics_data):
        """Load financial metrics into database"""
        # Get company_id
        company_query = "SELECT company_id FROM companies WHERE ticker = :ticker"
        result = self.execute_query(company_query, {'ticker': ticker})
        if not result:
            logger.warning(f"Company {ticker} not found in database")
            return
        
        company_id = result[0][0]
        
        query = """
        INSERT INTO financial_metrics (company_id, date, pe_ratio, pb_ratio, ps_ratio, debt_to_equity, 
                                     current_ratio, quick_ratio, roe, roa, profit_margin, revenue_growth, 
                                     earnings_growth, dividend_yield, payout_ratio)
        VALUES (:company_id, :date, :pe_ratio, :pb_ratio, :ps_ratio, :debt_to_equity, :current_ratio, :quick_ratio, :roe, :roa, :profit_margin, :revenue_growth, :earnings_growth, :dividend_yield, :payout_ratio)
        ON CONFLICT (company_id, date) DO UPDATE SET
            pe_ratio = EXCLUDED.pe_ratio,
            pb_ratio = EXCLUDED.pb_ratio,
            ps_ratio = EXCLUDED.ps_ratio,
            debt_to_equity = EXCLUDED.debt_to_equity,
            current_ratio = EXCLUDED.current_ratio,
            quick_ratio = EXCLUDED.quick_ratio,
            roe = EXCLUDED.roe,
            roa = EXCLUDED.roa,
            profit_margin = EXCLUDED.profit_margin,
            revenue_growth = EXCLUDED.revenue_growth,
            earnings_growth = EXCLUDED.earnings_growth,
            dividend_yield = EXCLUDED.dividend_yield,
            payout_ratio = EXCLUDED.payout_ratio
        """
        
        for date, row in metrics_data.iterrows():
            params = {
                'company_id': company_id,
                'date': date if isinstance(date, dt_date) else date.date(),
                'pe_ratio': self._convert_numpy_types(row.get('PE_Ratio')),
                'pb_ratio': self._convert_numpy_types(row.get('PB_Ratio')),
                'ps_ratio': self._convert_numpy_types(row.get('PS_Ratio')),
                'debt_to_equity': self._convert_numpy_types(row.get('Debt_to_Equity')),
                'current_ratio': self._convert_numpy_types(row.get('Current_Ratio')),
                'quick_ratio': self._convert_numpy_types(row.get('Quick_Ratio')),
                'roe': self._convert_numpy_types(row.get('ROE')),
                'roa': self._convert_numpy_types(row.get('ROA')),
                'profit_margin': self._convert_numpy_types(row.get('Profit_Margin')),
                'revenue_growth': self._convert_numpy_types(row.get('Revenue_Growth')),
                'earnings_growth': self._convert_numpy_types(row.get('Earnings_Growth')),
                'dividend_yield': self._convert_numpy_types(row.get('Dividend_Yield')),
                'payout_ratio': self._convert_numpy_types(row.get('Payout_Ratio'))
            }
            self.execute_query(query, params)
        
        logger.info(f"Loaded financial metrics for {ticker}")
    
    def get_stock_prices(self, ticker, start_date=None, end_date=None):
        """Retrieve stock prices for a company"""
        query = """
        SELECT sp.date, sp.open_price, sp.high_price, sp.low_price, sp.close_price, 
               sp.adjusted_close, sp.volume
        FROM stock_prices sp
        JOIN companies c ON sp.company_id = c.company_id
        WHERE c.ticker = :ticker
        """
        params = {'ticker': ticker}
        
        if start_date:
            query += " AND sp.date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query += " AND sp.date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY sp.date"
        
        result = self.execute_query(query, params)
        
        if result:
            # Convert Decimal types to float for analysis
            converted_result = []
            for row in result:
                converted_row = [
                    row[0],  # date
                    self._convert_decimal_types(row[1]),  # Open
                    self._convert_decimal_types(row[2]),  # High
                    self._convert_decimal_types(row[3]),  # Low
                    self._convert_decimal_types(row[4]),  # Close
                    self._convert_decimal_types(row[5]),  # Adj Close
                    self._convert_decimal_types(row[6])   # Volume
                ]
                converted_result.append(converted_row)
            
            df = pd.DataFrame(converted_result, columns=['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            df.set_index('date', inplace=True)
            return df
        return pd.DataFrame()
    
    def get_company_summary(self):
        """Get summary of all companies"""
        query = "SELECT * FROM company_summary ORDER BY market_cap DESC NULLS LAST"
        result = self.execute_query(query)
        
        if result:
            columns = ['company_id', 'ticker', 'company_name', 'sector', 'industry', 'market_cap',
                      'current_price', 'last_price_date', 'pe_ratio', 'pb_ratio', 'roe', 
                      'debt_to_equity', 'volatility', 'beta', 'sharpe_ratio']
            df = pd.DataFrame(result, columns=columns)
            return df
        return pd.DataFrame()
    
    def get_portfolio_analysis(self, date=None):
        """Get portfolio analysis data"""
        query = "SELECT * FROM portfolio_analysis"
        params = []
        
        if date:
            query += " WHERE date = %s"
            params.append(date)
        
        query += " ORDER BY date DESC, weight DESC"
        
        result = self.execute_query(query, tuple(params) if params else None)
        
        if result:
            columns = ['date', 'ticker', 'company_name', 'shares_owned', 'share_price', 
                      'market_value', 'weight', 'total_value', 'daily_return', 'cumulative_return']
            df = pd.DataFrame(result, columns=columns)
            return df
        return pd.DataFrame()
    
    def insert_company_summary(self, summary_data):
        """Insert company summary data"""
        query = """
        INSERT INTO companies (ticker, company_name, sector, industry, country, market_cap, description)
        VALUES (:ticker, :company_name, :sector, :industry, :country, :market_cap, :description)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            country = EXCLUDED.country,
            market_cap = EXCLUDED.market_cap,
            description = EXCLUDED.description,
            updated_at = CURRENT_TIMESTAMP
        """
        
        params = {
            'ticker': summary_data.get('ticker'),
            'company_name': summary_data.get('company_name'),
            'sector': summary_data.get('sector', 'Renewable Energy'),
            'industry': summary_data.get('industry', 'Renewable Energy'),
            'country': summary_data.get('country', 'Unknown'),
            'market_cap': summary_data.get('market_cap'),
            'description': summary_data.get('description', f"{summary_data.get('company_name')} - Renewable Energy Company")
        }
        
        self.execute_query(query, params)
        logger.info(f"Company summary inserted for {summary_data.get('ticker')}")
    
    def insert_stock_price(self, price_data):
        """Insert stock price data"""
        # Get company_id
        company_query = "SELECT company_id FROM companies WHERE ticker = :ticker"
        result = self.execute_query(company_query, {'ticker': price_data['ticker']})
        if not result:
            logger.warning(f"Company {price_data['ticker']} not found in database")
            return
        
        company_id = result[0][0]
        
        query = """
        INSERT INTO stock_prices (company_id, date, open_price, high_price, low_price, close_price, adjusted_close, volume)
        VALUES (:company_id, :date, :open_price, :high_price, :low_price, :close_price, :adjusted_close, :volume)
        ON CONFLICT (company_id, date) DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume
        """
        
        params = {
            'company_id': company_id,
            'date': price_data['date'] if isinstance(price_data['date'], dt_date) else price_data['date'].date(),
            'open_price': self._convert_numpy_types(price_data['open']),
            'high_price': self._convert_numpy_types(price_data['high']),
            'low_price': self._convert_numpy_types(price_data['low']),
            'close_price': self._convert_numpy_types(price_data['close']),
            'adjusted_close': self._convert_numpy_types(price_data.get('adjusted_close', price_data['close'])),
            'volume': self._convert_numpy_types(price_data['volume'])
        }
        
        self.execute_query(query, params)

def create_database():
    """Create database and tables"""
    try:
        # Connect to PostgreSQL server (without specifying database)
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database='postgres'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Database {DB_NAME} created successfully")
        
        cursor.close()
        conn.close()
        
        # Connect to the new database and create tables
        db_manager = DatabaseManager()
        if db_manager.connect():
            # Read and execute SQL file
            with open('sql/create_tables.sql', 'r') as file:
                sql_commands = file.read()
            
            # Split and execute commands more carefully
            commands = []
            current_command = ""
            
            for line in sql_commands.split('\n'):
                line = line.strip()
                if line.startswith('--') or not line:  # Skip comments and empty lines
                    continue
                current_command += line + " "
                if line.endswith(';'):
                    commands.append(current_command.strip())
                    current_command = ""
            
            # Execute each command
            for command in commands:
                if command:
                    try:
                        logger.info(f"Executing: {command[:50]}...")
                        db_manager.execute_query(command)
                    except Exception as e:
                        logger.warning(f"Command failed: {e}")
                        continue
            
            logger.info("Database tables created successfully")
            
            # Create indexes separately after all tables exist
            logger.info("Creating database indexes...")
            index_commands = [
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_company_date ON stock_prices(company_id, date);",
                "CREATE INDEX IF NOT EXISTS idx_financial_metrics_company_date ON financial_metrics(company_id, date);",
                "CREATE INDEX IF NOT EXISTS idx_technical_indicators_company_date ON technical_indicators(company_id, date);",
                "CREATE INDEX IF NOT EXISTS idx_risk_metrics_company_date ON risk_metrics(company_id, date);",
                "CREATE INDEX IF NOT EXISTS idx_news_articles_company_date ON news_articles(company_id, published_date);",
                "CREATE INDEX IF NOT EXISTS idx_portfolio_performance_date ON portfolio_performance(date);",
                "CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_company_date ON portfolio_holdings(company_id, date);"
            ]
            
            for index_cmd in index_commands:
                try:
                    db_manager.execute_query(index_cmd)
                except Exception as e:
                    logger.warning(f"Index creation failed (might already exist): {e}")
            
            logger.info("Database indexes created successfully")
            db_manager.disconnect()
            return True
        else:
            logger.error("Failed to connect to database after creation")
            return False
        
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        logger.warning("This could be due to:")
        logger.warning("1. PostgreSQL not running - start with: sudo systemctl start postgresql")
        logger.warning("2. Wrong credentials - update config.py with correct username/password")
        logger.warning("3. PostgreSQL not installed - install with: sudo apt-get install postgresql postgresql-contrib")
        logger.warning("4. User 'postgres' doesn't exist - create with: sudo -u postgres createuser --interactive")
        return False
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False

if __name__ == "__main__":
    # Test database connection
    create_database()
    db_manager = DatabaseManager()
    if db_manager.connect():
        print("Database connection test successful")
        db_manager.disconnect()
    else:
        print("Database connection test failed") 