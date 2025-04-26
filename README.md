# Advanced Data Analyzer

A powerful data analysis application built with Streamlit that uses AI to analyze and visualize data through natural language processing.

## Features

- Natural language Q&A with your data
- Automatic data type detection
- Domain-specific analysis (Finance, Marketing, HR, etc.)
- Interactive training system
- Automatic chart generation
- Development and Production environments

## Project Structure

```
.
├── config/
│   ├── dev_config.py    # Development configuration
│   └── prod_config.py   # Production configuration
├── src/
│   ├── dev/            # Development version
│   │   └── app.py
│   └── prod/           # Production version
│       └── app.py
├── run.py              # Script to run either dev or prod
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Prerequisites

- Python 3.8 or higher
- Git
- A Kluster AI API key
- Basic understanding of data analysis concepts

## Setup

1. Clone the repository:
```bash
git clone https://github.com/blkjack/dave.git
cd dave
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Development Environment
```bash
python run.py --env dev
```
The development environment includes:
- Debug mode for detailed error tracking
- Test data generation
- Higher credit limits for testing
- Detailed logging
- Development-specific features

### Production Environment
```bash
python run.py --env prod
```
The production environment includes:
- Performance optimizations
- Stricter error handling
- Production-specific logging
- Stricter credit limits
- Cached functions for better performance

## Configuration

### Development Configuration (config/dev_config.py)
- Debug mode enabled
- Test data available
- Higher credit limits
- Detailed logging
- Experimental features enabled

### Production Configuration (config/prod_config.py)
- Optimized for performance
- Stricter error handling
- Production-specific logging
- Stricter credit limits
- Experimental features disabled

## API Integration

The application uses the Kluster AI API for natural language processing. To use the application:

1. Obtain a Kluster AI API key
2. Enter the API key in the sidebar when running the application
3. Select the appropriate model for your needs

## Data Processing

The application supports:
- CSV file uploads (up to 30,000 rows in production)
- Automatic data type detection
- Domain-specific analysis
- Natural language query processing
- Automatic visualization generation

## Error Handling

The application includes comprehensive error handling:
- Development: Detailed error messages with stack traces
- Production: User-friendly error messages
- Logging of all errors for debugging
- Graceful fallbacks for common issues

## Contributing

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "Description of your changes"
```

3. Push to your branch:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add appropriate comments and docstrings
- Update documentation for new features
- Write tests for new functionality
- Ensure backward compatibility

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Verify your API key is correct
   - Check your internet connection
   - Ensure you have sufficient credits

2. **Data Loading Issues**
   - Verify CSV file format
   - Check file size limits
   - Ensure proper encoding

3. **Performance Issues**
   - Clear browser cache
   - Reduce dataset size
   - Check system resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- OpenAI for the AI capabilities
- All contributors who have helped improve this project

## Contact

For support or questions, please:
- Open an issue on GitHub
- Contact the maintainers
- Join our community discussions