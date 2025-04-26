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

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd advanced-data-analyzer
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

### Production Environment
```bash
python run.py --env prod
```

## Configuration

### Development Configuration
- Debug mode enabled
- Test data available
- Higher credit limits
- Detailed logging

### Production Configuration
- Optimized for performance
- Stricter error handling
- Production-specific logging
- Stricter credit limits

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- OpenAI for the AI capabilities
- All contributors who have helped improve this project