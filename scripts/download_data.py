"""
Enhanced script to download the IEEE-CIS Fraud Detection dataset from Kaggle.

Downloads both train and test datasets:
- train_transaction.csv.zip
- train_identity.csv
- test_transaction.csv
- test_identity.csv

Usage:
    python scripts/download_data.py [--output OUTPUT_DIR] [--api-key API_KEY_PATH]

Requirements:
    - Kaggle API credentials (typically stored in ~/.kaggle/kaggle.json)
    - Kaggle API Python package (pip install kaggle)
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Files to download from the competition
COMPETITION_FILES = [
    "train_transaction.csv.zip",  # Compressed due to large size
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv"
]

def setup_kaggle_credentials(api_key_path=None):
    """
    Set up Kaggle API credentials if not already configured.
    
    Args:
        api_key_path: Optional path to kaggle.json file
    
    Returns:
        bool: True if credentials are set up, False otherwise
    """
    # Default Kaggle credentials path
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Check if credentials already exist
    if kaggle_json.exists():
        logger.info("Kaggle credentials already exist.")
        return True
    
    # If api_key_path provided, use it
    if api_key_path and Path(api_key_path).exists():
        logger.info(f"Using provided API key from {api_key_path}")
        
        # Create .kaggle directory if it doesn't exist
        kaggle_dir.mkdir(exist_ok=True)
        
        # Copy file
        with open(api_key_path, 'r') as src:
            with open(kaggle_json, 'w') as dst:
                dst.write(src.read())
        
        # Set permissions
        os.chmod(kaggle_json, 0o600)
        logger.info("Kaggle API credentials installed successfully.")
        return True
    
    # If no credentials found, guide the user
    logger.error("Kaggle API credentials not found.")
    print("\nTo use the Kaggle API, you need to set up credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
    print("4. Run 'chmod 600 ~/.kaggle/kaggle.json' to set permissions\n")
    print("Alternatively, provide the path to your kaggle.json file using the --api-key argument.")
    
    return False

def check_kaggle_installed():
    """
    Check if the Kaggle API package is installed.
    
    Returns:
        bool: True if installed, False otherwise
    """
    try:
        import kaggle
        return True
    except ImportError:
        logger.error("Kaggle API package not installed.")
        print("\nPlease install the Kaggle API package:")
        print("pip install kaggle\n")
        return False

def download_ieee_cis_dataset(output_dir="./data/raw"):
    """
    Download the IEEE-CIS Fraud Detection dataset files from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset files
    
    Returns:
        bool: True if download successful, False otherwise
    """
    logger.info(f"Downloading IEEE-CIS Fraud Detection dataset to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    success = True
    
    # Download each file from the competition
    for filename in COMPETITION_FILES:
        try:
            logger.info(f"Downloading {filename}...")
            
            # Check if file already exists
            output_path = Path(output_dir) / filename
            if output_path.exists():
                logger.info(f"File {filename} already exists. Skipping download.")
                continue
                
            # Download using Kaggle API
            subprocess.run([
                "kaggle", "competitions", "download", 
                "-c", "ieee-fraud-detection", 
                "-f", filename, 
                "-p", output_dir
            ], check=True)
            
            logger.info(f"{filename} downloaded successfully.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
            success = False
    
    if success:
        logger.info("All dataset files downloaded successfully.")
    else:
        logger.warning("Some files failed to download. See log for details.")
    
    return success

def extract_transaction_data(output_dir="./data/raw"):
    """
    Extract the zipped transaction data file if present.
    
    Args:
        output_dir: Directory containing the dataset files
    
    Returns:
        bool: True if extraction successful or not needed, False otherwise
    """
    # Check for transaction zip file
    transaction_zip = Path(output_dir) / "train_transaction.csv.zip"
    transaction_csv = Path(output_dir) / "train_transaction.csv"
    
    # If already extracted, skip
    if transaction_csv.exists():
        logger.info("Transaction data already extracted.")
        return True
    
    # If zip exists, extract it
    if transaction_zip.exists():
        logger.info("Extracting transaction data...")
        try:
            subprocess.run(["unzip", "-o", transaction_zip, "-d", output_dir], check=True)
            logger.info("Extraction complete.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting zip file: {str(e)}")
            print(f"\nPlease extract the ZIP file manually:")
            print(f"unzip -o {transaction_zip} -d {output_dir}")
            return False
    
    return True  # No extraction needed

def main():
    """Main function to handle the download and extraction process."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download IEEE-CIS Fraud Detection dataset from Kaggle")
    parser.add_argument("--output", default="./data/raw", help="Output directory for dataset files")
    parser.add_argument("--api-key", help="Path to kaggle.json API key file")
    args = parser.parse_args()
    
    # Check and setup prerequisites
    if not check_kaggle_installed():
        return False
    
    if not setup_kaggle_credentials(args.api_key):
        return False
    
    # Download the dataset
    if not download_ieee_cis_dataset(args.output):
        return False
    
    # Extract transaction data if needed
    if not extract_transaction_data(args.output):
        return False
    
    print(f"\nDataset downloaded and extracted successfully to {args.output}")
    print("\nDataset files:")
    for file in Path(args.output).glob("*.*"):
        print(f"  - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    
    print("\nNext steps:")
    print("1. Run data preprocessing:   python -m doit downcast")
    print("2. Generate features:        python -m doit global_features customer_features")
    print("3. Train models:             python -m doit train_global_model train_customer_models")
    print("4. Evaluate on test data:    python -m doit evaluate_test")
    print("5. Or run the entire pipeline with: python -m doit")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)