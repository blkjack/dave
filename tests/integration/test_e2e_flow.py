"""
End-to-end tests for the complete application flow
"""
import unittest
import os
import sys
import pandas as pd
from pathlib import Path
import streamlit as st

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.dev.app import main
from src.dev.utils.data_utils import load_data, detect_dataset_type, get_data_summary
from src.dev.services.ai_service import AIService

class TestE2EFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Get API key from environment variable
        cls.api_key = os.getenv('KLUSTER_API_KEY')
        if not cls.api_key:
            raise ValueError("KLUSTER_API_KEY environment variable not set")
        
        # Load test data
        cls.test_data_path = Path(__file__).resolve().parent.parent.parent / 'test_data.csv'
        cls.df = load_data(open(cls.test_data_path))
        cls.dataset_type = detect_dataset_type(cls.df)
        cls.data_summary = get_data_summary(cls.df)
        
        # Initialize AI service
        cls.ai_service = AIService(
            api_key=cls.api_key,
            base_url="https://api.kluster.ai/v1",
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )

    def test_complete_flow(self):
        """Test the complete application flow"""
        # 1. Test data loading
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df), 5)
        self.assertEqual(list(self.df.columns), ['date', 'revenue', 'profit', 'expenses', 'campaign', 'clicks', 'conversions'])
        
        # 2. Test dataset type detection
        self.assertEqual(self.dataset_type, "Finance")
        
        # 3. Test data summary generation
        summary = get_data_summary(self.df)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        
        # 4. Test question generation
        questions = self.ai_service.generate_clarifying_questions(
            self.data_summary,
            self.dataset_type
        )
        self.assertEqual(len(questions), 5)
        
        # 5. Test query processing
        query = "What is the average revenue?"
        response = self.ai_service.process_query(
            query,
            "You are a data analyst",
            self.data_summary
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
        # 6. Test visualization generation
        query = "Show me the trend of revenue over time"
        response = self.ai_service.process_query(
            query,
            "You are a data analyst",
            self.data_summary
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_error_handling(self):
        """Test error handling in the complete flow"""
        # Test with invalid data
        with self.assertRaises(Exception):
            load_data(StringIO("invalid,csv,data"))
        
        # Test with invalid API key
        invalid_service = AIService(
            api_key="invalid_key",
            base_url="https://api.kluster.ai/v1",
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )
        
        # Should still work with default questions
        questions = invalid_service.generate_clarifying_questions(
            self.data_summary,
            self.dataset_type
        )
        self.assertEqual(len(questions), 5)

if __name__ == '__main__':
    unittest.main() 