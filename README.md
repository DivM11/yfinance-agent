# YFinance Agent

## Overview
The YFinance Agent is a Python-based application designed to help users build personalized finance portfolios using US equities. The application leverages the following technologies:

- **Streamlit**: For building an interactive and user-friendly dashboard.
- **Yfinance**: For retrieving stock market data.
- **OpenRouter**: For semantic matching and LLM inference to understand user input and provide intelligent recommendations.
- **Custom Model Support**: Allows integration of custom models for advanced analysis.

## Features
1. **User Input Handling**: Accepts natural language input to understand user preferences for portfolio creation.
2. **Stock Data Retrieval**: Fetches financial data from YFinance, including historical prices, financial statements, and analyst recommendations.
3. **Filtering and Analysis**: Applies user-defined filters and performs backtesting and forecasting.
4. **Interactive Dashboard**: Visualizes data and analysis results using Streamlit.

## Project Structure
```
yfinance-agent/
│
├── docs/               # Documentation files
├── config.yml          # Application configuration
├── main.py             # Entry point for the application
├── pyproject.toml      # Poetry configuration file
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
- Create a local `.env` file (gitignored) to store the OpenRouter API key.

Example `.env`:
```bash
OPENROUTER_API_KEY="your_api_key_here"
```

You can copy `.env.example` to get started:
```bash
cp .env.example .env
```

### OpenRouter Model Setup
- The app uses `qwen/qwen3-32b-04-28:nitro` for ticker generation, weight generation, and analysis.
- OpenRouter settings are grouped under `openrouter.api`, `openrouter.models`, `openrouter.outputs`, `openrouter.temperatures`, and `openrouter.prompts` in [config.yml](config.yml).
- Set the API key via environment variable name specified in `openrouter.api.key_env_var` (default: `OPENROUTER_API_KEY`).

## Using Docker

### Build the Docker Image
To build the Docker image for the YFinance Agent, run the following command:
```bash
docker build -t yfinance-agent .
```

### Run the Application in App Mode
To run the application in "app mode" (default), use the following command:
```bash
docker run -p 8501:8501 yfinance-agent
```

### Run the Application in Test Mode
To run the application in "test mode" (to execute the test suite), use the following command:
```bash
docker run -e APP_MODE=test yfinance-agent
```

### Docker Rebuild And Test Run

```bash
docker build -t yfinance-agent .
docker run --rm --name yfinance-agent-test yfinance-agent pytest
docker run --rm --name yfinance-agent-app -p 8501:8501 yfinance-agent
```

## Code Standards
This project follows:
- **PEP8**: Python style guide.
- **SOLID Principles**: For maintainable and scalable code.

## Contributing
Contributions are welcome! Please follow the code standards and submit a pull request.

## License
This project is licensed under the MIT License.