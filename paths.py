# paths.py

from pathlib import Path
import sys

# CONFIGURATION START (DREAMS or MASS)
SELECTED_DATASET = "MASS"

# Project root
ROOT_DIR = Path(__file__).parent

# Specific Data Paths
RAW_DREAMS_DATA_DIR = ROOT_DIR / "data" / "DREAMS"
RAW_MASS_DATA_DIR = ROOT_DIR / "data" / "MASS"

# Output directories
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "model_reports"
PLOTS_DIR = ROOT_DIR / "plots"

# DYNAMIC PATH ASSIGNMENT
if SELECTED_DATASET == "DREAMS":
    DATA_DIRECTORY = RAW_DREAMS_DATA_DIR
elif SELECTED_DATASET == "MASS":
    DATA_DIRECTORY = RAW_MASS_DATA_DIR
else:
    print(f"ERROR: Unknown dataset selected in paths.py: {SELECTED_DATASET}")
    sys.exit(1)

# Create output dirs if not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)