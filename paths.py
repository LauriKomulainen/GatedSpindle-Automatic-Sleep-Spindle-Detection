# paths.py
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent

# Path for DREAMS dataset
RAW_DREAMS_DATA_DIR = ROOT_DIR / "data" / "DREAMS"

# Path for MASS-SS2 dataset
RAW_MASS_DATA_DIR = ROOT_DIR / "data" / "MASS"

# Ouput directory for processed data
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Ouput directory for training reports
REPORTS_DIR = ROOT_DIR / "model_reports"

# Ouput directory for plots
PLOTS_DIR = ROOT_DIR / "plots"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)