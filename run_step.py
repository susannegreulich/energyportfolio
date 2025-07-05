#!/usr/bin/env python3
"""
Step Runner for Renewable Energy Investment Analysis
Allows running individual steps or ranges of steps for debugging and testing
"""

import subprocess
import sys
import logging
import os
import argparse

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define all available steps
ALL_STEPS = [
    ("setup", "setup.py", "Initial Setup and Configuration"),
    ("fetch", "fetch_data.py", "Data Fetching and Database Initialization"),
    ("analysis", "investment_analysis.py", "Investment Analysis and Technical Charts"),
    ("portfolio", "portfolio_optimization.py", "Portfolio Optimization Analysis"),
    ("presentation", "presentation.py", "Presentation and Report Generation")
]

def run_script(script_name, description):
    """Run a Python script and handle any errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"SCRIPT: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            logger.warning(f"Warnings from {script_name}: {result.stderr}")
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        print(f"❌ Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def list_steps():
    """List all available steps with their descriptions"""
    print("\nAvailable steps:")
    print("-" * 60)
    for i, (short_name, script, description) in enumerate(ALL_STEPS, 1):
        print(f"{i}. {short_name:12} - {description}")
        print(f"   Script: {script}")
    print("-" * 60)

def run_step(step_identifier):
    """Run a single step by name or number"""
    # Try to find step by number
    if step_identifier.isdigit():
        step_num = int(step_identifier) - 1
        if 0 <= step_num < len(ALL_STEPS):
            short_name, script, description = ALL_STEPS[step_num]
            return run_script(script, description)
        else:
            print(f"❌ Invalid step number: {step_identifier}")
            return False
    
    # Try to find step by short name
    for short_name, script, description in ALL_STEPS:
        if short_name.lower() == step_identifier.lower():
            return run_script(script, description)
    
    print(f"❌ Unknown step: {step_identifier}")
    return False

def run_range(start_step, end_step):
    """Run a range of steps"""
    try:
        start_idx = int(start_step) - 1
        end_idx = int(end_step) - 1
        
        if start_idx < 0 or end_idx >= len(ALL_STEPS) or start_idx > end_idx:
            print("❌ Invalid range specified")
            return False
        
        print(f"\n🚀 Running steps {start_idx + 1} to {end_idx + 1}")
        
        for i in range(start_idx, end_idx + 1):
            short_name, script, description = ALL_STEPS[i]
            success = run_script(script, description)
            if not success:
                print(f"\n❌ Pipeline failed at step {i + 1}: {description}")
                return False
        
        print(f"\n✅ Steps {start_idx + 1} to {end_idx + 1} completed successfully")
        return True
        
    except ValueError:
        print("❌ Invalid range format. Use numbers like '1-3'")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run individual steps or ranges of the Renewable Energy Investment Analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_step.py --list                    # List all available steps
  python run_step.py setup                     # Run setup step
  python run_step.py 1                         # Run step 1 (setup)
  python run_step.py fetch                     # Run fetch data step
  python run_step.py 2-4                       # Run steps 2, 3, and 4
  python run_step.py --range 1 3               # Run steps 1 to 3
  python run_step.py --all                     # Run all steps
        """
    )
    
    parser.add_argument('step', nargs='?', help='Step to run (name or number)')
    parser.add_argument('--list', action='store_true', help='List all available steps')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--range', nargs=2, metavar=('START', 'END'), 
                       help='Run range of steps (e.g., 1 3 for steps 1-3)')
    
    args = parser.parse_args()
    
    if args.list:
        list_steps()
        return
    
    if args.all:
        print("🚀 Running all steps...")
        for i, (short_name, script, description) in enumerate(ALL_STEPS, 1):
            success = run_script(script, description)
            if not success:
                print(f"\n❌ Pipeline failed at step {i}: {description}")
                sys.exit(1)
        print("\n🎉 All steps completed successfully!")
        return
    
    if args.range:
        success = run_range(args.range[0], args.range[1])
        if not success:
            sys.exit(1)
        return
    
    if args.step:
        success = run_step(args.step)
        if not success:
            sys.exit(1)
        return
    
    # No arguments provided, show help
    parser.print_help()
    print("\n💡 Quick reference:")
    list_steps()

if __name__ == "__main__":
    main() 