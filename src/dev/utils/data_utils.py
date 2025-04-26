"""
Utility functions for data processing and analysis
"""
import pandas as pd
import numpy as np
import logging
import time
import io

logger = logging.getLogger(__name__)

def load_data(uploaded_file, max_rows=1000, progress_bar=None):
    """
    Load and preprocess data from uploaded file
    
    Args:
        uploaded_file: The uploaded file object
        max_rows: Maximum number of rows to load
        progress_bar: Streamlit progress bar object
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        # Read the file content into memory
        content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        # Create a file-like object from the content
        file_obj = io.StringIO(content.decode('utf-8'))
        
        # Read the file in chunks to show progress
        chunks = []
        total_rows = 0
        
        # First pass to count rows
        for chunk in pd.read_csv(file_obj, chunksize=1000):
            total_rows += len(chunk)
            if total_rows > max_rows:
                break
        
        # Reset file pointer
        file_obj.seek(0)
        
        # Second pass to load data
        rows_loaded = 0
        for chunk in pd.read_csv(file_obj, chunksize=1000):
            if rows_loaded >= max_rows:
                break
                
            # Update progress
            if progress_bar:
                progress = min(rows_loaded / max_rows, 1.0)
                progress_bar.progress(progress, f"Loading data: {rows_loaded}/{max_rows} rows")
            
            chunks.append(chunk)
            rows_loaded += len(chunk)
            
            # Small delay to show progress
            time.sleep(0.1)
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        # Final progress update
        if progress_bar:
            progress_bar.progress(1.0, "Data loading complete!")
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    finally:
        # Clean up
        if 'file_obj' in locals():
            file_obj.close()

def detect_dataset_type(df):
    """
    Detect the type of dataset based on column names
    
    Args:
        df: Input dataframe
        
    Returns:
        str: Dataset type (Finance, Marketing, HR, etc.)
    """
    try:
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
    except Exception as e:
        logger.error(f"Error detecting dataset type: {str(e)}")
        return "Unknown"

def get_data_summary(df):
    """
    Generate a summary of the dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        str: Summary statistics
    """
    try:
        return df.describe(include='all').to_string()
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        return "Error generating summary"

def create_test_dataset():
    """
    Create a sample dataset for testing
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    try:
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
    except Exception as e:
        logger.error(f"Error creating test dataset: {str(e)}")
        return pd.DataFrame() 