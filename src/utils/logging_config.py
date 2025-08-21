import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "code_review.log", level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())

    # Create logs directory if it doesn't exist
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)