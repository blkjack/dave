"""
Production configuration settings
"""

# API Settings
API_BASE_URL = "https://api.kluster.ai/v1"
DEFAULT_MODEL = "anthropic.claude-3-opus-20240229"  # More stable model for production

# Data Settings
MAX_ROWS = 30000  # Full dataset size
DEBUG_MODE = False

# Feature Flags
ENABLE_ADVANCED_FEATURES = True
ENABLE_EXPERIMENTAL_FEATURES = False

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "prod.log"

# Production-specific settings
CREDITS_LIMIT = 100  # Stricter limit for production
ENABLE_TEST_DATA = False 