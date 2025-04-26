"""
Integration tests for API functionality
"""
import unittest
import os
import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.dev.services.ai_service import AIService
from src.dev.utils.data_utils import load_data, detect_dataset_type, get_data_summary

class TestAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and API key"""
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

    def test_question_generation(self):
        """Test generating clarifying questions"""
        questions = self.ai_service.generate_clarifying_questions(
            self.data_summary,
            self.dataset_type
        )
        
        # Verify questions
        self.assertEqual(len(questions), 5)
        self.assertTrue(all(isinstance(q, str) for q in questions))
        self.assertTrue(all('?' in q for q in questions))

    def test_query_processing(self):
        """Test processing a query"""
        # Test a simple query
        query = "What is the average revenue?"
        response = self.ai_service.process_query(
            query,
            "You are a data analyst",
            self.data_summary
        )
        
        # Verify response
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
        # Test a more complex query
        query = "Show me the trend of revenue over time"
        response = self.ai_service.process_query(
            query,
            "You are a data analyst",
            self.data_summary
        )
        
        # Verify response
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_error_handling(self):
        """Test error handling with invalid API key"""
        # Create service with invalid key
        invalid_service = AIService(
            api_key="invalid_key",
            base_url="https://api.kluster.ai/v1",
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )
        
        # Test question generation with invalid key
        questions = invalid_service.generate_clarifying_questions(
            self.data_summary,
            self.dataset_type
        )
        
        # Should return default questions
        self.assertEqual(len(questions), 5)
        self.assertTrue(all(isinstance(q, str) for q in questions))

if __name__ == '__main__':
    unittest.main() 