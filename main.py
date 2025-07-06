"""
Main Investment Analysis Script for Renewable Energy Companies
Orchestrates the complete analysis pipeline by calling individual scripts
"""

import subprocess
import sys
import logging
import os

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_run_setup():
    """Check if setup has been completed and run setup.py if needed"""
    print(f"\n{'='*60}")
    print("CHECKING SETUP PREREQUISITES")
    print(f"{'='*60}")
    
    # Check if config.py exists and has been configured
    config_exists = os.path.exists("config.py")
    
    if not config_exists:
        print("❌ config.py not found. Running setup...")
        return run_script("setup.py", "Initial Setup and Configuration")
    
    # Check if config.py has been properly configured (not just template)
    try:
        with open("config.py", "r") as f:
            config_content = f.read()
            if "your_finnhub_api_key_here" in config_content:
                print("⚠️  config.py appears to have default values. Running setup...")
                return run_script("setup.py", "Setup and Configuration")
    except Exception as e:
        print(f"❌ Error reading config.py: {e}. Running setup...")
        return run_script("setup.py", "Setup and Configuration")
    
    # Check if database connection works
    try:
        import psycopg2
        from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.close()
        print("✅ Database connection verified")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Running setup to fix database configuration...")
        return run_script("setup.py", "Database Setup and Configuration")
    
    print("✅ Setup prerequisites verified")
    return True

def run_script(script_name, description):
    """Run a Python script and handle any errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    try:
        # Use Popen for real-time output with unbuffered mode
        process = subprocess.Popen([sys.executable, "-u", script_name],  # -u flag forces unbuffered output
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                                 text=True, 
                                 bufsize=0,  # Unbuffered
                                 universal_newlines=True)
        
        # Print output in real-time with immediate flushing
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip(), flush=True)  # Force immediate output
        
        # Wait for process to complete
        return_code = process.poll()
        
        if return_code == 0:
            return True
        else:
            logger.error(f"Script {script_name} failed with return code {return_code}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main function that orchestrates the complete analysis pipeline"""
    print("Starting Renewable Energy Investment Analysis Pipeline")
    print("=" * 60)
    
    # Check and run setup if needed
    if not check_and_run_setup():
        print("\n❌ Setup failed. Please fix any issues and try again.")
        return False
    
    # Define the analysis steps in order
    steps = [
        ("fetch_data.py", "Data Fetching and Database Initialization"),
        ("investment_analysis.py", "Investment Analysis and Technical Charts"),
        ("portfolio_optimization.py", "Portfolio Optimization Analysis"),
        ("presentation.py", "Presentation and Report Generation")
    ]
    
    # Run each step
    for script, description in steps:
        success = run_script(script, description)
        if not success:
            print(f"\nPipeline failed at step: {description}")
            print("Please check the logs and fix any issues before continuing.")
            return False
    
    print(f"\n{'='*60}")
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nCheck the following directories for results:")
    print("• 'reports/' - Analysis reports")
    print("• 'charts/' - Technical analysis charts")
    print("• 'powerbi/data/' - CSV files for PowerBI dashboard")
    print("• 'data/' - Raw and processed data files")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 