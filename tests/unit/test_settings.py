"""
Unit tests for settings
"""
import unittest
import os
from pathlib import Path
from src.dev.config.settings import (
    API_BASE_URL,
    DEFAULT_MODEL,
    MAX_ROWS,
    DEBUG_MODE,
    ENABLE_TEST_DATA,
    SESSION_KEYS,
    DOMAIN_PROMPTS
)

class TestSettings(unittest.TestCase):
    def test_api_settings(self):
        """Test API settings"""
        self.assertEqual(API_BASE_URL, "https://api.kluster.ai/v1")
        self.assertEqual(DEFAULT_MODEL, "meta-llama/Llama-4-Scout-17B-16E-Instruct")

    def test_data_settings(self):
        """Test data settings"""
        self.assertEqual(MAX_ROWS, 1000)
        self.assertTrue(DEBUG_MODE)
        self.assertTrue(ENABLE_TEST_DATA)

    def test_session_keys(self):
        """Test session state keys"""
        # Verify all required keys are present
        required_keys = [
            'df', 'data_summary', 'history', 'understanding_score',
            'dataset_type', 'clarifying_questions', 'clarifying_answers',
            'active_tab', 'chat_history'
        ]
        for key in required_keys:
            self.assertIn(key, SESSION_KEYS)
        
        # Verify default values
        self.assertIsNone(SESSION_KEYS['df'])
        self.assertIsNone(SESSION_KEYS['data_summary'])
        self.assertEqual(SESSION_KEYS['history'], [])
        self.assertEqual(SESSION_KEYS['understanding_score'], 0)
        self.assertEqual(SESSION_KEYS['dataset_type'], "Unknown")
        self.assertEqual(SESSION_KEYS['clarifying_questions'], [])
        self.assertEqual(SESSION_KEYS['clarifying_answers'], {})
        self.assertEqual(SESSION_KEYS['active_tab'], "Training")
        self.assertEqual(SESSION_KEYS['chat_history'], [])

    def test_domain_prompts(self):
        """Test domain-specific prompts"""
        # Verify all domains are present
        domains = ["Finance", "Marketing", "HR", "Healthcare", "Education"]
        for domain in domains:
            self.assertIn(domain, DOMAIN_PROMPTS)
        
        # Verify prompt content
        for domain, prompt in DOMAIN_PROMPTS.items():
            self.assertIsInstance(prompt, str)
            self.assertTrue(len(prompt) > 0)
            self.assertIn("Special instructions for", prompt)

if __name__ == '__main__':
    unittest.main() 