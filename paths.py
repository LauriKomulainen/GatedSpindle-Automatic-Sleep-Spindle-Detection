# paths.py

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent

# Raw data paths (each dataset has its own build/run scripts)
RAW_DREAMS_DATA_DIR = ROOT_DIR / "data" / "DREAMS"
RAW_MASS_DATA_DIR = ROOT_DIR / "data" / "MASS"

# Output directories
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "model_reports"
PLOTS_DIR = ROOT_DIR / "plots"

# Create output dirs if not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)