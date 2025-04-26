"""
Development configuration settings
"""

# API Settings
API_BASE_URL = "https://api.kluster.ai/v1"
DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Data Settings
MAX_ROWS = 1000  # Smaller dataset for development
DEBUG_MODE = True

# Feature Flags
ENABLE_ADVANCED_FEATURES = True
ENABLE_EXPERIMENTAL_FEATURES = True

# Logging
LOG_LEVEL = "DEBUG"
LOG_FILE = "dev.log"

# Development-specific settings
CREDITS_LIMIT = 1000  # Higher limit for development
ENABLE_TEST_DATA = True 