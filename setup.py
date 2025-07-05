#!/usr/bin/env python3
"""
Enhanced Setup script for Renewable Energy Investment Analysis Project
- Checks/installs PostgreSQL
- Ensures service is running
- Creates/updates DB user and database
- Updates config.py with working credentials
- Guides the user through the process
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import getpass
import random
import string
import time

# --- Helper Functions ---
def run_cmd(cmd, capture_output=False, check=False, shell=False):
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=shell, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=shell, check=check)
            return None
    except subprocess.CalledProcessError as e:
        return None

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    directories = [
        "data/raw",
        "analysis",
        "reports",
        "charts",
        "powerbi/data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def check_postgresql_installed():
    print("\n🗄️ Checking PostgreSQL installation...")
    result = run_cmd(["which", "psql"], capture_output=True)
    if result:
        print("✅ PostgreSQL is installed")
        return True
    else:
        print("❌ PostgreSQL is not installed.")
        if platform.system() == "Linux":
            print("Installing PostgreSQL (requires sudo password)...")
            run_cmd("sudo apt update && sudo apt install postgresql postgresql-contrib -y", shell=True)
            result = run_cmd(["which", "psql"], capture_output=True)
            if result:
                print("✅ PostgreSQL installed successfully")
                return True
            else:
                print("❌ Failed to install PostgreSQL. Please install it manually.")
                return False
        else:
            print("Please install PostgreSQL manually for your OS.")
            return False

def check_postgresql_service():
    print("\n🔄 Checking PostgreSQL service status...")
    status = run_cmd(["sudo", "systemctl", "is-active", "postgresql"], capture_output=True)
    if status == "active":
        print("✅ PostgreSQL service is running")
        return True
    else:
        print("PostgreSQL service is not running. Attempting to start...")
        run_cmd(["sudo", "systemctl", "start", "postgresql"])
        time.sleep(2)
        status = run_cmd(["sudo", "systemctl", "is-active", "postgresql"], capture_output=True)
        if status == "active":
            print("✅ PostgreSQL service started")
            return True
        else:
            print("❌ Failed to start PostgreSQL service. Please start it manually.")
            return False

def random_password(length=12):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def check_create_db_user(user, password):
    print(f"\n👤 Checking if PostgreSQL user '{user}' exists...")
    check_user_cmd = f"sudo -u postgres psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='{user}';\""
    exists = run_cmd(check_user_cmd, capture_output=True, shell=True)
    if exists.strip() == '1':
        print(f"✅ User '{user}' exists.")
        return True
    else:
        print(f"User '{user}' does not exist. Creating...")
        create_user_cmd = f"sudo -u postgres psql -c \"CREATE USER {user} WITH PASSWORD '{password}';\""
        run_cmd(create_user_cmd, shell=True)
        print(f"✅ User '{user}' created with password '{password}'")
        # Grant CREATEDB privilege
        run_cmd(f"sudo -u postgres psql -c \"ALTER USER {user} CREATEDB;\"", shell=True)
        return True

def check_create_database(dbname, user):
    print(f"\n🗄️ Checking if database '{dbname}' exists...")
    check_db_cmd = f"sudo -u postgres psql -tAc \"SELECT 1 FROM pg_database WHERE datname='{dbname}';\""
    exists = run_cmd(check_db_cmd, capture_output=True, shell=True)
    if exists.strip() == '1':
        print(f"✅ Database '{dbname}' exists.")
        return True
    else:
        print(f"Database '{dbname}' does not exist. Creating...")
        create_db_cmd = f"sudo -u postgres createdb {dbname} -O {user}"
        run_cmd(create_db_cmd, shell=True)
        print(f"✅ Database '{dbname}' created and owned by '{user}'")
        return True

def update_config_py(db_host, db_port, db_name, db_user, db_password):
    print("\n📝 Updating config.py with database credentials...")
    config_path = "config.py"
    with open(config_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("DB_HOST"):
            new_lines.append(f'DB_HOST = "{db_host}"
')
        elif line.strip().startswith("DB_PORT"):
            new_lines.append(f'DB_PORT = {db_port}
')
        elif line.strip().startswith("DB_NAME"):
            new_lines.append(f'DB_NAME = "{db_name}"
')
        elif line.strip().startswith("DB_USER"):
            new_lines.append(f'DB_USER = "{db_user}"
')
        elif line.strip().startswith("DB_PASSWORD"):
            new_lines.append(f'DB_PASSWORD = "{db_password}"
')
        else:
            new_lines.append(line)
    with open(config_path, "w") as f:
        f.writelines(new_lines)
    print("✅ config.py updated.")

def create_config_template():
    """Create config template if it doesn't exist"""
    if not os.path.exists("config.py"):
        print("\n⚙️ Creating config.py template...")
        config_content = '''# Configuration file for Renewable Energy Investment Analysis
# Copy this file to config.py and add your actual API keys

# API Configuration
# Get your free API key from https://finnhub.io/
FINNHUB_API_KEY = "your_finnhub_api_key_here"

# Alpha Vantage API for additional financial data
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"

# Data configuration
DATA_START_DATE = "2018-01-01"
DATA_END_DATE = "2025-01-01"
NEWS_DAYS_BACK = 30
MAX_NEWS_ARTICLES_PER_COMPANY = 10

# Analysis Configuration
RISK_FREE_RATE = 0.02  # 2% risk-free rate (approximate 10-year Treasury yield)
MARKET_RETURN = 0.10   # 10% expected market return
BENCHMARK_INDEX = "^GSPC"  # S&P 500 as benchmark

# Portfolio Configuration
INITIAL_INVESTMENT = 100000  # $100,000 initial investment
REBALANCE_FREQUENCY = "Q"    # Quarterly rebalancing

# Database Configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "energy_investment_db"
DB_USER = "energy_user"
DB_PASSWORD = "energy123"

# Analysis Parameters
VOLATILITY_WINDOW = 252  # 1 year for volatility calculation
BETA_WINDOW = 252        # 1 year for beta calculation
CORRELATION_WINDOW = 252 # 1 year for correlation calculation
SHARPE_RATIO_WINDOW = 252 # 1 year for Sharpe ratio calculation

# Technical Analysis Parameters
SMA_PERIODS = [20, 50, 200]  # Simple Moving Average periods
RSI_PERIOD = 14              # RSI calculation period
MACD_FAST = 12              # MACD fast period
MACD_SLOW = 26              # MACD slow period
MACD_SIGNAL = 9             # MACD signal period

# Fundamental Analysis Parameters
PE_RATIO_THRESHOLD = 25     # P/E ratio threshold for value stocks
DEBT_TO_EQUITY_THRESHOLD = 0.5  # Debt-to-equity threshold
ROE_THRESHOLD = 0.15        # Return on Equity threshold
GROWTH_THRESHOLD = 0.10     # Revenue growth threshold
'''
        with open("config.py", "w") as f:
            f.write(config_content)
        print("✅ Created config.py template")
        print("⚠️  Please update config.py with your database credentials and API keys")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    required_modules = [
        "pandas",
        "numpy",
        "yfinance",
        "matplotlib",
        "seaborn",
        "scipy",
        "psycopg2",
        "sqlalchemy",
        "textblob"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All required modules imported successfully")
        return True

# --- Main Setup Logic ---
def main():
    """Main setup function"""
    print("🚀 Renewable Energy Investment Analysis Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check PostgreSQL installation
    if not check_postgresql_installed():
        sys.exit(1)
    
    # Check PostgreSQL service
    if not check_postgresql_service():
        sys.exit(1)
    
    # Create config template if needed
    create_config_template()
    
    # --- PostgreSQL Credentials Setup ---
    db_host = "localhost"
    db_port = 5432
    db_name = "energy_investment_db"
    db_user = "energy_user"
    db_password = "energy123"
    
    # Check if user wants to use default or custom credentials
    print("\nDefault database user: energy_user")
    print("Default password: energy123")
    print("Default database: energy_investment_db")
    resp = input("Do you want to use these defaults? (Y/n): ").strip().lower()
    if resp == 'n':
        db_user = input("Enter database username: ").strip()
        db_password = getpass.getpass("Enter database password (leave blank to generate random): ")
        if not db_password:
            db_password = random_password()
            print(f"Generated password: {db_password}")
        db_name = input("Enter database name: ").strip()
        if not db_name:
            db_name = "energy_investment_db"
    
    # Check/create user and database
    check_create_db_user(db_user, db_password)
    check_create_database(db_name, db_user)
    
    # Update config.py
    update_config_py(db_host, db_port, db_name, db_user, db_password)
    
    # Test imports
    if not test_imports():
        print("\n❌ Some imports failed. Please check the installation.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update config.py with your API keys if needed")
    print("2. Ensure PostgreSQL is running and accessible")
    print("3. Run: python main.py")
    print("4. For PowerBI: python powerbi_export.py")
    
    print("\n📚 For more information, see README.md")
    print(f"\n🔑 Your database credentials are:\n  Host: {db_host}\n  Port: {db_port}\n  Database: {db_name}\n  User: {db_user}\n  Password: {db_password}")

if __name__ == "__main__":
    main() 