# utils/logger.py

import logging
import sys
import os

LOG_DIR = "logs"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(log_file_name: str):
    """
    Configures logging for the entire project.
    Creates the 'logs' directory if it doesn't exist.
    Writes to the specified log_file_name.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory {LOG_DIR}: {e}")

    # Dynamic path
    log_file_path = os.path.join(LOG_DIR, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    log = logging.getLogger(__name__)
    log.info(f"Logger configured successfully. Logging to {log_file_path}")