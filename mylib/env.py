import os
import logging
import deepxde as dde


# library path
DIR_LIB = os.path.dirname(os.path.abspath(__file__))
# project base path (parent of library path)
DIR_BASE = os.path.abspath(os.path.join(DIR_LIB, os.pardir))
# experiments path
DIR_EXPERIMENTS = os.path.join(DIR_BASE, "experiments")
# analysis path
DIR_ANALYSIS = os.path.join(DIR_BASE, "analysis")
# datasets path
DIR_DATASETS = os.path.join(DIR_BASE, "datasets")

# Define log format
LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logging level

# Create a formatter
log_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)  # Set format for console logs
logger.addHandler(console_handler)  # Add console handler to logger


# Set float64 as default float point type
dde.config.set_default_float("float64")
