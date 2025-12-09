# AI Trading Arena - ML Stock Prediction System

A comprehensive AI-powered stock trading and prediction system that combines traditional machine learning (XGBoost) with LLM-based trading agents (Claude, GPT, DeepSeek) to analyze and predict stock prices for TSX (Toronto Stock Exchange) listed companies.

## Features

- **Multi-Model Prediction**: XGBoost-based models with 40+ technical indicators
- **LLM Trading Agents**: Claude, GPT-4, and DeepSeek agents for market analysis
- **Interactive Dashboard**: Real-time visualization of predictions and model performance
- **TSX Market Focus**: Specialized for top 10 TSX stocks (banks, energy, technology)
- **Full-Stack Architecture**: Python backend with React/TypeScript frontend
- **GPU Acceleration**: CUDA-enabled XGBoost training

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                React Dashboard (Port 3000)              │
│    Model Metrics | Feature Importance | Predictions    │
└─────────────────────────────────────────────────────────┘
                        ↕ REST API + WebSockets
┌─────────────────────────────────────────────────────────┐
│              Flask Backend (Port 8000)                  │
│    Training API | Predictions | Analytics | Agents     │
└─────────────────────────────────────────────────────────┘
                        ↕
┌──────────────────┬──────────────────┬──────────────────┐
│   SQLite DBs     │   XGBoost Models │   LLM Agents     │
│  stocks.db       │   agentVD        │   Claude/GPT     │
│  model_results   │   Returns        │   DeepSeek       │
└──────────────────┴──────────────────┴──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- GPU with CUDA support (optional, for faster training)

### Installation

1. **Clone and navigate to project:**
```bash
cd c:\Projects2025\MLProject\Project
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install frontend dependencies:**
```bash
cd src/dashboard
npm install
cd ../..
```

5. **Configure environment variables:**

Create a `.env` file in the project root:
```bash
ALPHA_VANTAGE_KEY=your_key_here
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-proj-...
DEEPSEEK_API_KEY=sk-...
```

6. **Initialize data and train models:**
```bash
python runner.py
```

This will:
- Fetch 5 years of historical data for TSX top 10 stocks
- Clean and process the data
- Engineer 40+ technical indicators
- Train XGBoost models for each symbol
- Save results to `model_results.db`

7. **Start the application:**
```bash
python start_dashboard.py
```

The dashboard will open at `http://localhost:3000`

## Project Structure

```
Project/
├── app.py                    # Flask backend entry point
├── runner.py                 # Data pipeline & training script
├── start_dashboard.py        # Full-stack launcher
├── config.toml               # Configuration settings
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── agents/
│   │   ├── custom_agent/     # agentVD (XGBoost models)
│   │   ├── llm_agents/       # Claude, GPT, DeepSeek agents
│   │   └── agentsconfig/     # Base agent classes
│   ├── backend/              # Flask API server
│   │   └── routes/           # API endpoints
│   ├── config/               # Config management
│   ├── dashboard/            # React frontend
│   ├── database/             # Database utilities
│   ├── evaluation/           # Model metrics
│   ├── processing/           # Data cleaning pipeline
│   └── util/                 # Logging utilities
│
├── data/
│   ├── fetchers/             # Data acquisition scripts
│   └── storage/
│       └── stocks.db         # Historical OHLCV data
│
├── models/                   # Trained XGBoost models (JSON)
├── model_results.db          # Training results & metrics
└── logs/                     # Application logs
```

## Usage

### Training Models

Run the full training pipeline:
```bash
python runner.py
```

### Starting the Backend Only

```bash
python app.py
```

Backend API will be available at `http://localhost:8000`

### Starting the Full Stack

```bash
python start_dashboard.py
```

Or use the batch file (Windows):
```bash
start_dashboard.bat
```

## API Endpoints

### Models
- `POST /api/models/train` - Train a new model
- `GET /api/models/predict` - Get prediction for symbol
- `GET /api/models/feature-importance` - Get feature importance

### Analytics
- `GET /api/analytics/performance` - Model performance metrics
- `GET /api/analytics/portfolio` - Portfolio analytics

### Agents
- `GET /api/agents/list` - List available agents
- `POST /api/agents/analyze` - Get agent analysis for symbol

### Symbols
- `GET /api/symbol/<symbol>` - Get symbol data and predictions

## Supported Stocks (TSX Top 10)

### Banks
- RY.TO - Royal Bank of Canada
- TD.TO - Toronto-Dominion Bank
- BNS.TO - Bank of Nova Scotia
- BMO.TO - Bank of Montreal
- CM.TO - Canadian Imperial Bank of Commerce

### Energy
- ENB.TO - Enbridge Inc.
- CNQ.TO - Canadian Natural Resources
- SU.TO - Suncor Energy
- TRP.TO - TC Energy

### Technology
- SHOP.TO - Shopify Inc.

## Machine Learning Features

### Technical Indicators (40+)

**Returns & Momentum**
- 1, 3, 5, 10, 20-day returns
- 20-day momentum

**Moving Averages**
- SMA: 5, 20, 60-day
- EMA: 12, 26-day
- Price-to-SMA ratios

**Volatility**
- 5, 20, 60-day rolling volatility
- ATR (Average True Range)
- Bollinger Bands (width, %B)

**Momentum Indicators**
- MACD and signal line
- RSI (7 and 14-day)
- Distance from 20-day high/low

**Volume Indicators**
- 20-day average volume
- Volume spike ratio
- Volume z-score
- OBV (On-Balance Volume)

**Time Features**
- Day of week
- Month

### Model Configuration

**Sector-Specific Hyperparameters:**

```python
# Banks (conservative, stable)
n_estimators=1800, max_depth=4, learning_rate=0.02

# Energy (volatile, complex)
n_estimators=2200, max_depth=5, learning_rate=0.018

# Technology (high growth)
n_estimators=2000, max_depth=5, learning_rate=0.025
```

**Training Strategy:**
- Temporal split: 70% train / 15% validation / 15% test
- Early stopping on validation set
- GPU acceleration (CUDA)
- Feature shift = 1 day (prevent data leakage)

## LLM Agents

### Claude Agent
- Model: Claude Sonnet 4.5
- Temperature: 0.7
- Max tokens: 1000
- Cost tracking enabled

### GPT Agent
- Model: GPT-4o
- Temperature: 0.7
- Max tokens: 1000
- Cost tracking enabled

### DeepSeek Agent
- Model: DeepSeek Reasoner
- Temperature: 1.0
- Max tokens: 8000
- Advanced reasoning capabilities

### Agent Features
- Technical analysis interpretation
- Sentiment scoring
- Confidence-based position sizing
- Entry/exit/stop-loss recommendations
- Risk factor identification

## Evaluation Metrics

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **MDAE** - Median Absolute Error
- **R²** - R-squared score
- **MAPE** - Mean Absolute Percentage Error
- **Direction Accuracy** - Percentage of correct up/down predictions

## Configuration

Edit `config.toml` to customize:

- API rate limits and keys
- Database paths and settings
- LLM model parameters (temperature, max tokens)
- Agent behavior (position sizing, stop-loss, take-profit)
- Cost limits (daily/monthly)
- Cache settings

## Development

### Backend Development

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Flask in debug mode
python app.py
```

### Frontend Development

```bash
cd src/dashboard

# Start development server
npm start

# Build for production
npm run build
```

### Running Tests

```bash
# Run Python tests (if available)
pytest

# Run frontend tests
cd src/dashboard
npm test
```

## Database Schema

### stocks.db
- `candle` - Raw OHLCV data
- `structured_data` - Engineered features

### model_results.db
- `models` - Model metadata
- `training_results` - Performance metrics
- `predictions` - Individual predictions
- `input_features` - Features per prediction
- `feature_importance` - Feature rankings

## Troubleshooting

### Import Errors
Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Database Errors
Reset and reinitialize:
```bash
python runner.py
```

### Frontend Won't Start
Install dependencies:
```bash
cd src/dashboard
npm install
```

### API Connection Issues
Check that backend is running on port 8000:
```bash
curl http://localhost:8000/api/health
```

## Performance Tips

- **GPU Training**: Ensure CUDA is installed for faster model training
- **Data Caching**: Enable database caching in `config.toml`
- **Batch Processing**: Train multiple symbols using `UnifiedMultiSymbolTrainer`
- **API Rate Limits**: Configure rate limits in `config.toml` to avoid throttling

## Security Notes

- Never commit `.env` file (contains API keys)
- Database files (`*.db`) are gitignored
- API keys should have minimal required permissions
- Set cost limits for LLM API usage

## License

This project is for educational and research purposes.

## Acknowledgments

- XGBoost for gradient boosting framework
- Anthropic for Claude API
- OpenAI for GPT API
- DeepSeek for reasoning models
- yfinance for stock data

## Support

For issues and questions:
- Check the logs in `logs/` directory
- Review `config.toml` settings
- Ensure all API keys are valid in `.env`

---

**Built with Python, React, XGBoost, and Claude Code**
