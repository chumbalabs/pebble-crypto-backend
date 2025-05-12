![Banner Image](static\images\20250109_094420_0000.png )

# pebble-crypto-api
This is the backend service for the **Pebble Crypto** app, a cryptocurrency signals platform built using **FastAPI**.

Advanced cryptocurrency prediction API with real-time analysis and AI-powered insights.

## Features ✨

### Core Features
- 📈 Real-time price predictions with confidence scoring
- 🔍 Technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR)
- 📊 Order book analytics (support/resistance levels, buy/sell walls)
- 🤖 AI-powered market analysis with Gemini
- 🗣️ Natural language query system for crypto market data
- 🌡️ Market health monitoring (volatility, liquidity)

### Technical Features
- ⚡ Async-first architecture for high concurrency
- 🧩 Modular design with focused components
- 🔒 Rate limiting (10 RPM per endpoint)
- 🧠 Smart caching (OHLCV data, predictions)
- 📊 Built-in metrics tracking
- 🛡️ Error resilience with automatic retries
- 🐳 Docker support for easy deployment

## Environment Setup ⚙️
The application uses a `.env` file for configuration, which is already set up. The environment variables include:

```ini
# API Configuration
BINANCE_API=https://api.binance.com/api/v3
GEMINI_API_KEY=your_gemini_key_here
CACHE_TTL=300  # 5 minutes

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
WORKERS=1

# Security
ALLOWED_ORIGINS=*
API_RATE_LIMIT=100/hour
METRICS_INTERVAL=300  # 5 minutes
```

## Project Structure 📁
```text
pebble-crypto-backend/
├── app/                      # Main application package
│   ├── api/                  # API interface
│   │   └── routes/           # Route definitions
│   │       ├── ai_agent.py   # Natural language query endpoint
│   │       ├── health.py     # Health check endpoint
│   │       ├── market_data.py # Market data endpoints
│   │       ├── predictions.py # Prediction endpoints
│   │       └── websockets.py  # WebSocket handlers
│   ├── core/                 # Core business logic
│   │   ├── ai/               # AI components
│   │   │   ├── agent.py      # AI agent orchestration
│   │   │   └── gemini_client.py # Gemini integration
│   │   ├── indicators/       # Technical indicators
│   │   │   ├── advanced/     # Advanced indicators (Bollinger, ATR)
│   │   │   └── order_book/   # Order book analytics
│   │   └── prediction/       # Prediction models
│   │       └── technical.py  # Technical analysis models
│   ├── services/             # External services
│   │   ├── binance.py        # Binance API client
│   │   └── metrics.py        # Performance tracking
│   └── main.py               # FastAPI entry point
├── static/                   # Static assets
├── main.py                   # Legacy entry point
├── .env                      # Environment configuration
├── Dockerfile                # Docker image definition
└── docker-compose.yml        # Docker Compose configuration
```

## API Endpoints 📡
| Endpoint          | Method | Description                     | Rate Limit   |
|-------------------|--------|---------------------------------|--------------|
| `/api/health`     | GET    | API health check                | 100/min      |
| `/predict/{symbol}` | GET    | Price prediction + AI analysis  | 30/min       |
| `/symbols`        | GET    | Active trading pairs            | 30/min       |
| `/intraday/{symbol}` | GET  | Intraday data with custom intervals | 30/min   |
| `/historical/{symbol}` | GET | Historical data with custom intervals | 20/min |
| `/ws/realtime/{symbol}` | WS | Real-time price streaming with custom intervals | - |
| `/api/ask`        | POST   | Natural language query API      | 10/min       |

## Natural Language Queries 🗣️
The new AI agent feature allows you to ask questions in natural language:

```http
POST /api/ask
Content-Type: application/json

{
  "question": "What is the current price of BTC?"
}
```

Example queries:
- "What is the price of Ethereum right now?"
- "What's the trend for BTC over the last day?"
- "Is ADA volatile today?"
- "Should I buy or sell SOL?"
- "Are there any buy walls for BNB?"

## Supported Timeframes ⏰
The API supports the following timeframes for data retrieval and analysis:

- **Hours**: 1h, 2h, 4h, 6h, 8h, 12h
- **Days**: 1d, 3d
- **Weeks**: 1w
- **Months**: 1M

Use these interval values with the `/predict`, `/intraday`, and `/historical` endpoints.

## Rate Limits ⏱️
- Global limit: 100 requests/hour
- Prediction endpoint: 30 requests/minute
- AI queries: 10 requests/minute
- Symbols endpoint: 30 requests/minute
- Exceeding limits returns `429 Too Many Requests`

## Error Handling ❗
Standard error response format:
```json
{
  "error": "Error Type",
  "detail": "Human-readable description",
  "timestamp": "ISO-8601 datetime"
}
```

## Development 🛠️
```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload --port 8000

# For legacy version
uvicorn main:app --reload --port 8000
```

## Docker Deployment 🐳
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Building and Running with Docker Manually
```bash
# Build the Docker image
docker build -t pebble-crypto-api .

# Run the container
docker run -d -p 8000:8000 --env-file .env --name pebble-crypto-api pebble-crypto-api
```

## Testing 🔍
```bash
# Get BTC prediction with 1-hour interval
curl "http://localhost:8000/predict/BTCUSDT?interval=1h"

# Get historical data with 4-hour interval
curl "http://localhost:8000/historical/BTCUSDT?interval=4h&limit=50"

# Ask a natural language question
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the price of BTC?"}'

# Stream real-time data with 1-hour interval
wscat -c "ws://localhost:8000/ws/realtime/BTCUSDT?interval=1h"
```

## License 📄
MIT License - See [LICENSE](LICENSE) for details

> **Note**  
> This is not financial advice. Cryptocurrency trading carries significant risk.