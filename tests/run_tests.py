"""
Test runner script
"""
import unittest
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_tests():
    """Run all tests"""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 