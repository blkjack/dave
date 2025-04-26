"""
Unit tests for data utilities
"""
import unittest
import pandas as pd
import numpy as np
from io import StringIO
from src.dev.utils.data_utils import (
    load_data,
    detect_dataset_type,
    get_data_summary,
    create_test_dataset
)

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a sample CSV string
        self.sample_csv = """date,value,category
2023-01-01,100,A
2023-01-02,150,B
2023-01-03,200,C"""
        
        # Create a sample dataframe
        self.sample_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3),
            'value': [100, 150, 200],
            'category': ['A', 'B', 'C']
        })
        
        # Create a finance dataset
        self.finance_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3),
            'revenue': [1000, 1500, 2000],
            'profit': [100, 150, 200],
            'expenses': [900, 1350, 1800]
        })
        
        # Create a marketing dataset
        self.marketing_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3),
            'campaign': ['A', 'B', 'C'],
            'clicks': [100, 150, 200],
            'conversions': [10, 15, 20]
        })

    def test_load_data(self):
        """Test loading data from CSV"""
        # Create a file-like object from the CSV string
        csv_file = StringIO(self.sample_csv)
        
        # Test loading data
        df = load_data(csv_file)
        
        # Verify the data was loaded correctly
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['date', 'value', 'category'])
        self.assertEqual(df['value'].dtype, 'float64')

    def test_detect_dataset_type(self):
        """Test dataset type detection"""
        # Test finance dataset
        self.assertEqual(detect_dataset_type(self.finance_df), "Finance")
        
        # Test marketing dataset
        self.assertEqual(detect_dataset_type(self.marketing_df), "Marketing")
        
        # Test general dataset
        self.assertEqual(detect_dataset_type(self.sample_df), "General")

    def test_get_data_summary(self):
        """Test data summary generation"""
        summary = get_data_summary(self.sample_df)
        
        # Verify summary contains expected information
        self.assertIn("count", summary)
        self.assertIn("mean", summary)
        self.assertIn("std", summary)

    def test_create_test_dataset(self):
        """Test test dataset creation"""
        test_df = create_test_dataset()
        
        # Verify test dataset structure
        self.assertEqual(len(test_df), 100)
        self.assertEqual(list(test_df.columns), ['date', 'value', 'category'])
        self.assertEqual(test_df['value'].dtype, 'float64')
        self.assertTrue(all(cat in ['A', 'B', 'C'] for cat in test_df['category']))

if __name__ == '__main__':
    unittest.main() 