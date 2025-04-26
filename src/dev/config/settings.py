"""
Configuration settings for the development environment
"""
import os
import logging
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = BASE_DIR / 'config'

# API Settings
API_BASE_URL = "https://api.kluster.ai/v1"
DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
API_KEY = "f25804e2-e893-41cc-859a-56f70511604c"  # Development API key

# Data Settings
MAX_ROWS = 1000  # Smaller dataset for development
DEBUG_MODE = True

# Feature Flags
ENABLE_ADVANCED_FEATURES = True
ENABLE_EXPERIMENTAL_FEATURES = True

# Logging
LOG_LEVEL = "DEBUG"
LOG_FILE = BASE_DIR / "dev.log"

# Development-specific settings
CREDITS_LIMIT = 1000  # Higher limit for development
ENABLE_TEST_DATA = True

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE
)

# Session state keys
SESSION_KEYS = {
    'df': None,
    'data_summary': None,
    'history': [],
    'understanding_score': 0,
    'dataset_type': "Unknown",
    'clarifying_questions': [],
    'clarifying_answers': {},
    'active_tab': "Training",
    'chat_history': []
}

# Domain-specific prompts
DOMAIN_PROMPTS = {
    "Finance": """
    Special instructions for Finance data:
    - When analyzing financial metrics, consider YoY growth and margins
    - Express monetary values with appropriate currency symbols
    - Focus on ROI, profitability, and financial performance metrics
    - Consider seasonality in financial data
    """,
    "Marketing": """
    Special instructions for Marketing data:
    - Focus on conversion rates, CAC, CLV, and ROI metrics
    - Consider campaign performance and audience segmentation
    - Look for correlations between marketing efforts and outcomes
    - Provide actionable marketing insights
    """,
    "HR": """
    Special instructions for HR data:
    - Focus on employee retention, satisfaction, and performance metrics
    - Consider departmental differences and team dynamics
    - Analyze compensation equity and promotion patterns
    - Look for factors that affect recruitment and turnover
    """,
    "Healthcare": """
    Special instructions for Healthcare data:
    - Focus on patient outcomes, treatment efficacy, and care quality
    - Consider demographic factors in health outcomes
    - Analyze resource utilization and operational efficiency
    - Be mindful of privacy considerations in analysis
    """,
    "Education": """
    Special instructions for Education data:
    - Focus on student performance, learning outcomes, and engagement
    - Consider demographic factors in educational outcomes
    - Analyze teaching effectiveness and resource allocation
    - Look for patterns in student achievement and growth
    """
} 