"""
Utility functions for data processing and analysis
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_data(uploaded_file, max_rows=1000):
    """
    Load and preprocess data from uploaded file
    
    Args:
        uploaded_file: The uploaded file object
        max_rows: Maximum number of rows to load
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        df = pd.read_csv(uploaded_file, dtype=str, low_memory=True, nrows=max_rows)
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
                
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def detect_dataset_type(df):
    """
    Detect the type of dataset based on column names
    
    Args:
        df: Input dataframe
        
    Returns:
        str: Dataset type (Finance, Marketing, HR, etc.)
    """
    cols = [col.lower() for col in df.columns]
    col_string = " ".join(cols)
    
    if any(term in col_string for term in ['price', 'cost', 'revenue', 'profit', 'expense', 'budget', 'sales']):
        return "Finance"
    elif any(term in col_string for term in ['campaign', 'customer', 'click', 'conversion', 'ctr', 'roi', 'lead']):
        return "Marketing"
    elif any(term in col_string for term in ['employee', 'salary', 'hire', 'performance', 'department', 'manager']):
        return "HR"
    elif any(term in col_string for term in ['patient', 'diagnosis', 'treatment', 'doctor', 'hospital', 'medication']):
        return "Healthcare"
    elif any(term in col_string for term in ['student', 'grade', 'course', 'class', 'teacher', 'school']):
        return "Education"
    else:
        return "General"

def get_data_summary(df):
    """
    Generate a summary of the dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        str: Summary statistics
    """
    return df.describe(include='all').to_string()

def create_test_dataset():
    """
    Create a sample dataset for testing
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }) 