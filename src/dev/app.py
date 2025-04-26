import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import time
import altair as alt
from openai import OpenAI
import re
from datetime import datetime
import sys
import os
import logging

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
from dev_config import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Data Analyzer (DEV)",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'understanding_score' not in st.session_state:
    st.session_state.understanding_score = 0
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = "Unknown"
if 'clarifying_questions' not in st.session_state:
    st.session_state.clarifying_questions = []
if 'clarifying_answers' not in st.session_state:
    st.session_state.clarifying_answers = {}
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Training"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App title and description
st.title("🔧 Advanced Data Analyzer (Development)")
st.markdown("""
This is the development version of the Advanced Data Analyzer.
Features and changes are being tested here before being deployed to production.
""")

# Development-specific features
if DEBUG_MODE:
    st.sidebar.header("Development Tools")
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.json({
            "Session State": st.session_state,
            "Config": {k: v for k, v in globals().items() if k.isupper()},
            "Environment": os.environ.get("ENVIRONMENT", "development")
        })

# Rest of your existing app code, but using config values
# For example:
# MAX_ROWS instead of hardcoded 30000
# DEFAULT_MODEL instead of hardcoded model name
# etc.

# Add development-specific features
if ENABLE_TEST_DATA:
    st.sidebar.header("Test Data")
    if st.sidebar.button("Load Test Dataset"):
        # Load a sample dataset for testing
        test_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        st.session_state.df = test_data
        st.success("Test dataset loaded!")

# Add development-specific logging
def log_operation(operation, details):
    if DEBUG_MODE:
        logger.debug(f"Operation: {operation}, Details: {details}")

# Modify your existing functions to use logging
def process_query(client, query, system_prompt, df_summary, chat_history=None):
    log_operation("process_query", {"query": query, "history_length": len(chat_history) if chat_history else 0})
    # Rest of your existing process_query function
    pass

# Add development-specific error handling
def safe_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if DEBUG_MODE:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                st.error(f"Development Error: {str(e)}")
            else:
                st.error("An error occurred. Please try again.")
    return wrapper

# Apply the decorator to key functions
process_query = safe_operation(process_query)
detect_dataset_type = safe_operation(detect_dataset_type)
calculate_understanding = safe_operation(calculate_understanding)

# Rest of your existing app code... 