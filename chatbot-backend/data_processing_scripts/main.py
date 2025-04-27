import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_processing.log')
    ]
)

logger = logging.getLogger(__name__)

def load_script(script_path: str) -> Any:
    """Load a Python script as a module"""
    try:
        spec = importlib.util.spec_from_file_location("script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading script {script_path}: {str(e)}")
        raise

def run_script(script_path: str) -> bool:
    """Run a Python script and return success status"""
    try:
        logger.info(f"Starting {script_path}...")
        module = load_script(script_path)
        module.main()
        logger.info(f"Completed {script_path} successfully")
        return True
    except Exception as e:
        logger.error(f"Error in {script_path}:")
        logger.error(f"Traceback:\n{str(e)}")
        return False

def main():
    """Main function to run all data processing scripts"""
    try:
        logger.info("Starting data processing pipeline...")
        
        # Get the directory containing this script
        current_dir = Path(__file__).parent
        
        # Define the order of scripts to run
        scripts = [
            "exploratory_data_analysis.py",
            "data_visualization.py",
            "modeling.py",
        ]
        
        # Run each script in sequence
        for script in scripts:
            script_path = current_dir / script
            if not script_path.exists():
                logger.error(f"Script not found: {script}")
                continue
                
            success = run_script(str(script_path))
            if not success:
                logger.error(f"Pipeline failed at {script}")
                return
        
        logger.info("Data processing pipeline completed successfully!")
        
    except Exception as e:
        logger.error("Data processing pipeline failed!")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 