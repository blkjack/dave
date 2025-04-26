"""
Unit tests for AI service
"""
import unittest
from unittest.mock import Mock, patch
import json
from src.dev.services.ai_service import AIService

class TestAIService(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.api_key = "test_api_key"
        self.base_url = "https://api.test.com/v1"
        self.model = "test-model"
        self.df_summary = "Sample data summary"
        self.dataset_type = "Finance"
        
        # Create a mock response for chat completions
        self.mock_response = Mock()
        self.mock_response.choices = [
            Mock(
                message=Mock(
                    content='["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]'
                )
            )
        ]
        
        # Create a mock client
        self.mock_client = Mock()
        self.mock_client.chat.completions.create.return_value = self.mock_response
        
        # Create the AI service with mocked client
        with patch('openai.OpenAI', return_value=self.mock_client):
            self.ai_service = AIService(self.api_key, self.base_url, self.model)

    def test_generate_clarifying_questions(self):
        """Test generating clarifying questions"""
        # Test successful question generation
        questions = self.ai_service.generate_clarifying_questions(
            self.df_summary,
            self.dataset_type
        )
        
        # Verify questions were generated
        self.assertEqual(len(questions), 5)
        self.assertTrue(all(isinstance(q, str) for q in questions))
        
        # Verify API was called correctly
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args['model'], self.model)
        self.assertEqual(call_args['max_completion_tokens'], 500)
        self.assertEqual(call_args['temperature'], 0.1)

    def test_process_query(self):
        """Test processing a query"""
        # Set up test data
        query = "What is the average revenue?"
        system_prompt = "You are a data analyst"
        chat_history = [
            {"user": "Previous question", "assistant": "Previous answer"}
        ]
        
        # Mock the planning response
        planning_response = Mock()
        planning_response.choices = [
            Mock(
                message=Mock(
                    content='{"query_type": "statistical_analysis", "columns_needed": ["revenue"], "analysis_steps": ["step1"], "visualization_needed": false, "visualization_type": "none"}'
                )
            )
        ]
        
        # Mock the execution response
        execution_response = Mock()
        execution_response.choices = [
            Mock(
                message=Mock(
                    content="The average revenue is $1,500"
                )
            )
        ]
        
        # Set up the mock to return different responses for planning and execution
        self.mock_client.chat.completions.create.side_effect = [
            planning_response,
            execution_response
        ]
        
        # Test query processing
        response = self.ai_service.process_query(
            query,
            system_prompt,
            self.df_summary,
            chat_history
        )
        
        # Verify response
        self.assertEqual(response, "The average revenue is $1,500")
        
        # Verify API was called twice (planning and execution)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)

    def test_error_handling(self):
        """Test error handling"""
        # Test error in question generation
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Should return default questions
        questions = self.ai_service.generate_clarifying_questions(
            self.df_summary,
            self.dataset_type
        )
        
        # Verify default questions were returned
        self.assertEqual(len(questions), 5)
        self.assertTrue(all(isinstance(q, str) for q in questions))
        
        # Test error in query processing
        with self.assertRaises(Exception):
            self.ai_service.process_query(
                "test query",
                "test prompt",
                "test summary"
            )

if __name__ == '__main__':
    unittest.main() 