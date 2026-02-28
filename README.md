# YFinance Agent

## Overview
The YFinance Agent is a Python-based application designed to help users build personalized finance portfolios using US equities. The application leverages the following technologies:

- **Streamlit**: For building an interactive and user-friendly dashboard.
- **Massive.com (formerly Polygon.io)**: For retrieving stock market data (OHLCV prices, SEC-sourced financial statements).
- **OpenRouter**: For semantic matching and LLM inference to understand user input and provide intelligent analysis.
- **Custom Model Support**: Allows integration of custom models for advanced analysis.

## Features
1. **User Input Handling**: Accepts natural language input to understand user preferences for portfolio creation.
2. **Stock Data Retrieval**: Fetches financial data from Massive.com, including historical prices and financial statements (income statement, balance sheet, cash flow).
3. **Filtering and Analysis**: Applies user-defined filters and performs backtesting and forecasting.
4. **Interactive Dashboard**: Visualizes data and analysis results using Streamlit.

## Project Structure
```
yfinance-agent/
│
├── docs/               # Documentation files
├── config.yml          # Application configuration
├── docker-compose.yml  # Docker Compose services
├── .secrets.example    # Template for .secrets (gitignored)
├── main.py             # Entry point for the application
├── pyproject.toml      # Poetry configuration file
├── src/
│   ├── config.py       # Configuration loading
│   ├── dashboard.py    # Streamlit dashboard UI
│   ├── data_client.py  # Massive.com (Polygon.io) data fetching
│   ├── llm_validation.py # LLM output validation
│   ├── plots.py        # Plotly chart builders
│   ├── portfolio.py    # Portfolio allocation
│   └── summaries.py    # Data summarization
├── tests/              # Test suite
└── README.md           # Project overview and instructions
```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd yfinance-agent
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```

### Running the Application
To start the Streamlit dashboard, run:
```bash
poetry run streamlit run main.py
```

## Configuration and Secrets
- Update [config.yml](config.yml) for model, prompts, and UI text.
- API keys are supplied as **environment variables** at runtime.

### Setting up secrets
Copy the example file and fill in your keys:
```bash
cp .secrets.example .secrets
```
Edit `.secrets`:
```
OPENROUTER_API_KEY=your_openrouter_key_here
MASSIVE_API_KEY=your_massive_com_key_here
```
> `.secrets` is git-ignored.

**Locally (shell):**
```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
export MASSIVE_API_KEY="your_massive_com_key_here"
poetry run streamlit run main.py
```

**Docker Compose (recommended):**
```bash
docker compose up # reads .secrets automatically
```

**Docker CLI (env-file):**
```bash
docker run -p 8501:8501 --env-file .secrets yfinance-agent
```

### Massive.com (Polygon.io) Setup
- **API key**: Sign up at [massive.com](https://massive.com) and obtain an API key.
- **Plan requirement**: The **Advanced plan ($199/mo)** is required for financial statement data (income statement, balance sheet). OHLCV price data is available on the free tier.
- The API key is loaded from the environment variable specified in `massive.api.key_env_var` (default: `MASSIVE_API_KEY`).
- Python SDK: `massive` (PyPI) — `pip install -U massive`

### OpenRouter Model Setup
- The app uses `qwen/qwen3-32b-04-28:nitro` for ticker generation, weight generation, and analysis.
- OpenRouter settings are grouped under `openrouter.api`, `openrouter.models`, `openrouter.outputs`, `openrouter.temperatures`, and `openrouter.prompts` in [config.yml](config.yml).
- Set the API key via environment variable name specified in `openrouter.api.key_env_var` (default: `OPENROUTER_API_KEY`).

## Using Docker

### Build the Docker Image
```bash
docker build -t yfinance-agent .
```

### Run with Docker Compose (recommended)
```bash
docker compose up                           # app on :8501
docker compose --profile test up            # run tests
```

### Run with Docker CLI
```bash
# App mode
docker run -p 8501:8501 --env-file .secrets yfinance-agent

# Test mode
docker run --rm --env-file .secrets yfinance-agent pytest
```

### Full rebuild cycle
```bash
docker compose build
docker compose --profile test run --rm test
docker compose up
```

## Code Standards
This project follows:
- **PEP8**: Python style guide.
- **SOLID Principles**: For maintainable and scalable code.

## Contributing
Contributions are welcome! Please follow the code standards and submit a pull request.

## License
This project is licensed under the MIT License.