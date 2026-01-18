# utils/logger.py

import logging
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent  # .../Sleep Spindle Detector/
LOG_DIR = PROJECT_ROOT / "logs"

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(log_file_name: str):
    """
    Configures logging for the entire project.
    Creates the 'logs' directory in the PROJECT ROOT if it doesn't exist.
    """
    try:
        # Luodaan logs-kansio absoluuttiseen polkuun
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory {LOG_DIR}: {e}")

    # Käsittely: Jos annettu nimi on jo koko polku, käytetään sitä.
    # Jos se on pelkkä nimi (esim. "plots.log"), yhdistetään se LOG_DIRiin.
    target_path = Path(log_file_name)
    if not target_path.is_absolute():
        log_file_path = LOG_DIR / log_file_name
    else:
        log_file_path = target_path

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(str(log_file_path), mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    log = logging.getLogger(__name__)
    log.info(f"Logger configured successfully. Logging to {log_file_path}")