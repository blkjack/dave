# Advanced Data Analyzer

A Streamlit-based application for advanced data analysis and visualization.

## Features

- Natural language Q&A with your data
- Interactive data visualization
- Automatic data type detection
- Domain-specific analysis
- Cost-effective response formatting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/blkjack/dave.git
cd dave
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
dave/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
└── .env               # Environment variables (not tracked in git)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License