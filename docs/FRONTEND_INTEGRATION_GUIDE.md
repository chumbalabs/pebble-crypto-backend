# Frontend Integration Guide

## 🎯 **Overview**

This guide provides comprehensive instructions for frontend developers building user interfaces for the Pebble Crypto Backend. The API has been extensively tested and verified to work with real market data from 5+ cryptocurrency exchanges.

> **⚠️ IMPORTANT FOR FRONTEND DEVELOPERS**
> 
> **This API is production-ready** with 100% test success rate across all endpoints. However, always verify current functionality before implementing, as schemas may evolve.

## 🚀 **Quick Start for Frontend**

### 1. Verify API is Running
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# Expected response:
# {"status": "healthy", "timestamp": "2025-01-09T12:34:56Z"}
```

### 2. Test Primary AI Endpoint
```bash
# Test the main feature - AI natural language queries
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Bitcoin doing today?"}'
```

### 3. Get Available Trading Symbols
```bash
# Get list of 1,452+ available trading pairs
curl "http://localhost:8000/api/symbols"
```

## 🎨 **UI/UX Recommendations**

### Primary Features to Implement

#### 1. **AI Chat Interface** (Primary Feature)
- **Input**: Text field for natural language queries
- **Output**: Formatted response with query classification
- **Examples**: "What is BTC price?", "Compare BTC and ETH", "Analyze my portfolio"
- **Rate Limit**: 60 queries/minute (show countdown timer)

#### 2. **Multi-Asset Dashboard**
- **Grid Layout**: Display multiple cryptocurrencies simultaneously
- **Real-time Updates**: Use WebSocket or polling for live data
- **Sorting**: By price, volume, change percentage
- **Filtering**: By exchange, market cap, volatility

#### 3. **Exchange Health Monitor**
- **Status Indicators**: Green/yellow/red for each exchange
- **Response Times**: Show latency for each exchange
- **Coverage Stats**: Display trading pair counts per exchange

#### 4. **Price Comparison Tool**
- **Cross-Exchange Prices**: Show prices from all 5 exchanges
- **Arbitrage Opportunities**: Highlight price differences
- **Best Price Highlighting**: Mark best bid/ask prices

## 📱 **Component Architecture**

### React Component Structure
```
src/
├── components/
│   ├── AIChat/
│   │   ├── ChatInput.jsx
│   │   ├── ChatResponse.jsx
│   │   └── QueryTypeIndicator.jsx
│   ├── Dashboard/
│   │   ├── AssetGrid.jsx
│   │   ├── AssetCard.jsx
│   │   └── PriceChart.jsx
│   ├── Exchange/
│   │   ├── HealthMonitor.jsx
│   │   ├── PriceComparison.jsx
│   │   └── ArbitrageAlert.jsx
│   └── Common/
│       ├── LoadingSpinner.jsx
│       ├── ErrorBoundary.jsx
│       └── RateLimitIndicator.jsx
├── hooks/
│   ├── useAPI.js
│   ├── useWebSocket.js
│   └── useRateLimit.js
├── services/
│   ├── api.js
│   └── websocket.js
└── utils/
    ├── formatters.js
    └── validators.js
```

## 🔌 **API Integration**

### Core API Service
```javascript
// services/api.js
class PebbleCryptoAPI {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.rateLimits = {
      ai: { limit: 60, window: 60000, requests: [] },
      market: { limit: 30, window: 60000, requests: [] },
      health: { limit: 100, window: 60000, requests: [] }
    };
  }

  // Rate limiting helper
  checkRateLimit(type) {
    const now = Date.now();
    const limit = this.rateLimits[type];
    
    // Remove old requests outside the window
    limit.requests = limit.requests.filter(time => now - time < limit.window);
    
    if (limit.requests.length >= limit.limit) {
      const oldestRequest = Math.min(...limit.requests);
      const waitTime = limit.window - (now - oldestRequest);
      throw new Error(`Rate limit exceeded. Wait ${Math.ceil(waitTime / 1000)} seconds`);
    }
    
    limit.requests.push(now);
  }

  // AI Natural Language Queries
  async askQuery(query) {
    this.checkRateLimit('ai');
    
    const response = await fetch(`${this.baseURL}/api/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Query failed');
    }
    
    return await response.json();
  }

  // Market Data
  async getSymbols() {
    this.checkRateLimit('health');
    const response = await fetch(`${this.baseURL}/api/symbols`);
    return await response.json();
  }

  async getPrediction(symbol, interval = '1h') {
    this.checkRateLimit('market');
    const response = await fetch(`${this.baseURL}/api/predict/${symbol}?interval=${interval}`);
    return await response.json();
  }

  async getHistoricalData(symbol, interval = '4h', limit = 100) {
    this.checkRateLimit('market');
    const response = await fetch(`${this.baseURL}/api/historical/${symbol}?interval=${interval}&limit=${limit}`);
    return await response.json();
  }

  async getInvestmentAdvice(symbol) {
    this.checkRateLimit('market');
    const response = await fetch(`${this.baseURL}/api/investment-advice/${symbol}`);
    return await response.json();
  }

  async compareAssets(symbols) {
    this.checkRateLimit('market');
    const symbolsParam = symbols.join(',');
    const response = await fetch(`${this.baseURL}/api/compare?symbols=${symbolsParam}`);
    return await response.json();
  }

  // Exchange Features
  async getExchangeHealth() {
    this.checkRateLimit('health');
    const response = await fetch(`${this.baseURL}/api/exchanges/health`);
    return await response.json();
  }

  async getBestPrices(symbol) {
    this.checkRateLimit('market');
    const response = await fetch(`${this.baseURL}/api/exchanges/best-prices/${symbol}`);
    return await response.json();
  }

  async getExchangeCoverage() {
    this.checkRateLimit('health');
    const response = await fetch(`${this.baseURL}/api/exchanges/coverage`);
    return await response.json();
  }
}

export default new PebbleCryptoAPI();
```

### React Hooks
```javascript
// hooks/useAPI.js
import { useState, useEffect } from 'react';
import api from '../services/api';

export function useAIQuery() {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const askQuery = async (query) => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await api.askQuery(query);
      setResponse(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { response, loading, error, askQuery };
}

export function useExchangeHealth() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const result = await api.getExchangeHealth();
        setHealth(result);
      } catch (error) {
        console.error('Failed to fetch exchange health:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  return { health, loading };
}

export function useSymbols() {
  const [symbols, setSymbols] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const result = await api.getSymbols();
        setSymbols(result.symbols || []);
      } catch (error) {
        console.error('Failed to fetch symbols:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSymbols();
  }, []);

  return { symbols, loading };
}
```

## 🎨 **Component Examples**

### AI Chat Component
```jsx
// components/AIChat/AIChat.jsx
import React, { useState } from 'react';
import { useAIQuery } from '../../hooks/useAPI';

function AIChat() {
  const [query, setQuery] = useState('');
  const { response, loading, error, askQuery } = useAIQuery();

  const handleSubmit = (e) => {
    e.preventDefault();
    askQuery(query);
    setQuery('');
  };

  const exampleQueries = [
    "What is Bitcoin's price?",
    "Compare BTC, ETH, and SOL",
    "How volatile is the crypto market today?",
    "Should I buy or sell Bitcoin?",
    "Find arbitrage opportunities for MATIC"
  ];

  return (
    <div className="ai-chat">
      <div className="chat-header">
        <h2>🤖 AI Crypto Analyst</h2>
        <p>Ask questions in natural language</p>
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about crypto prices, trends, or analysis..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? '🔄' : '📤'}
        </button>
      </form>

      <div className="example-queries">
        <p>Try these examples:</p>
        {exampleQueries.map((example, index) => (
          <button
            key={index}
            onClick={() => setQuery(example)}
            className="example-button"
          >
            {example}
          </button>
        ))}
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {response && (
        <div className="chat-response">
          <div className="response-header">
            <span className="query-type">
              {response.query_info?.query_type || 'analysis'}
            </span>
            <span className="symbols">
              {response.query_info?.symbols?.join(', ')}
            </span>
          </div>
          <div className="response-content">
            {response.response.split('\n').map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default AIChat;
```

### Exchange Health Monitor
```jsx
// components/Exchange/HealthMonitor.jsx
import React from 'react';
import { useExchangeHealth } from '../../hooks/useAPI';

function HealthMonitor() {
  const { health, loading } = useExchangeHealth();

  if (loading) return <div>Loading exchange status...</div>;
  if (!health) return <div>Unable to load exchange status</div>;

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return '#22c55e';
      case 'degraded': return '#f59e0b';
      case 'down': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="health-monitor">
      <h3>Exchange Status</h3>
      <div className="overall-status">
        <span className={`status-badge ${health.overall_status}`}>
          {health.overall_status.toUpperCase()}
        </span>
        <span className="status-count">
          {health.healthy_count}/{health.total_count} Healthy
        </span>
      </div>

      <div className="exchange-grid">
        {Object.entries(health.exchanges).map(([name, status]) => (
          <div key={name} className="exchange-card">
            <div className="exchange-header">
              <span className="exchange-name">{name}</span>
              <div 
                className="status-indicator"
                style={{ backgroundColor: getStatusColor(status.status) }}
              />
            </div>
            <div className="exchange-details">
              <div className="response-time">
                {status.response_time}ms
              </div>
              <div className="last-check">
                {new Date(status.last_check).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default HealthMonitor;
```

### Multi-Asset Dashboard
```jsx
// components/Dashboard/AssetDashboard.jsx
import React, { useState, useEffect } from 'react';
import api from '../../services/api';

function AssetDashboard() {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAssets, setSelectedAssets] = useState(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']);

  useEffect(() => {
    const fetchAssetData = async () => {
      setLoading(true);
      try {
        const promises = selectedAssets.map(symbol => 
          api.getPrediction(symbol).catch(err => ({ symbol, error: err.message }))
        );
        const results = await Promise.all(promises);
        setAssets(results);
      } catch (error) {
        console.error('Failed to fetch asset data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAssetData();
    const interval = setInterval(fetchAssetData, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, [selectedAssets]);

  const addAsset = (symbol) => {
    if (!selectedAssets.includes(symbol)) {
      setSelectedAssets([...selectedAssets, symbol]);
    }
  };

  const removeAsset = (symbol) => {
    setSelectedAssets(selectedAssets.filter(s => s !== symbol));
  };

  if (loading) return <div>Loading dashboard...</div>;

  return (
    <div className="asset-dashboard">
      <div className="dashboard-header">
        <h2>📊 Multi-Asset Dashboard</h2>
        <div className="asset-controls">
          <input
            type="text"
            placeholder="Add symbol (e.g., ADAUSDT)"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                addAsset(e.target.value.toUpperCase());
                e.target.value = '';
              }
            }}
          />
        </div>
      </div>

      <div className="asset-grid">
        {assets.map((asset, index) => (
          <div key={asset.symbol || index} className="asset-card">
            {asset.error ? (
              <div className="asset-error">
                <h3>{asset.symbol}</h3>
                <p>❌ {asset.error}</p>
                <button onClick={() => removeAsset(asset.symbol)}>
                  Remove
                </button>
              </div>
            ) : (
              <div className="asset-content">
                <div className="asset-header">
                  <h3>{asset.symbol}</h3>
                  <button onClick={() => removeAsset(asset.symbol)}>
                    ✕
                  </button>
                </div>
                
                <div className="price-info">
                  <div className="current-price">
                    ${asset.current_price?.toLocaleString()}
                  </div>
                  <div className={`price-change ${asset.prediction?.direction}`}>
                    {asset.prediction?.direction === 'bullish' ? '📈' : '📉'}
                    {asset.prediction?.direction}
                  </div>
                </div>

                <div className="prediction-info">
                  <div className="predicted-price">
                    Target: ${asset.prediction?.price?.toLocaleString()}
                  </div>
                  <div className="confidence">
                    Confidence: {(asset.prediction?.confidence * 100)?.toFixed(1)}%
                  </div>
                </div>

                <div className="technical-indicators">
                  <div className="indicator">
                    RSI: {asset.technical_indicators?.rsi}
                  </div>
                  <div className="indicator">
                    MACD: {asset.technical_indicators?.macd}
                  </div>
                </div>

                {asset.ai_analysis && (
                  <div className="ai-analysis">
                    <p>{asset.ai_analysis.substring(0, 100)}...</p>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default AssetDashboard;
```

## 🎯 **Best Practices**

### 1. Error Handling
```javascript
// Robust error handling with user-friendly messages
const handleAPIError = (error) => {
  if (error.message.includes('Rate limit')) {
    return 'Too many requests. Please wait a moment and try again.';
  } else if (error.message.includes('Exchange')) {
    return 'Exchange temporarily unavailable. Using backup data.';
  } else if (error.message.includes('Network')) {
    return 'Connection issue. Please check your internet connection.';
  } else {
    return 'Something went wrong. Please try again.';
  }
};
```

### 2. Loading States
```jsx
// Provide clear loading feedback
function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <div className="loading-spinner">
      <div className="spinner" />
      <p>{message}</p>
    </div>
  );
}

// Usage in components
{loading && <LoadingSpinner message="Analyzing crypto data..." />}
```

### 3. Rate Limit Indicators
```jsx
// Show rate limit status to users
function RateLimitIndicator({ type, used, limit }) {
  const percentage = (used / limit) * 100;
  const color = percentage > 80 ? 'red' : percentage > 60 ? 'orange' : 'green';
  
  return (
    <div className="rate-limit-indicator">
      <div className="rate-limit-bar">
        <div 
          className="rate-limit-fill"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
      <span>{used}/{limit} {type} queries</span>
    </div>
  );
}
```

### 4. Responsive Design
```css
/* Mobile-first responsive design */
.asset-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

@media (min-width: 768px) {
  .asset-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 1024px) {
  .asset-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1440px) {
  .asset-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

## 🔧 **Development Workflow**

### 1. Setup Development Environment
```bash
# Start backend server
uvicorn main:app --reload --port 8000

# In another terminal, start frontend dev server
npm start  # or yarn start
```

### 2. Testing Integration
```javascript
// Create test utilities
// utils/testHelpers.js
export const mockAPIResponse = (data) => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve(data),
    })
  );
};

// Test components with mocked API
import { render, screen, waitFor } from '@testing-library/react';
import { mockAPIResponse } from '../utils/testHelpers';
import AIChat from '../components/AIChat/AIChat';

test('AI chat displays response', async () => {
  mockAPIResponse({
    response: 'Bitcoin is trading at $105,000',
    query_info: { query_type: 'single_asset' }
  });

  render(<AIChat />);
  
  // Test interaction
  fireEvent.change(screen.getByPlaceholderText(/ask about crypto/i), {
    target: { value: 'What is BTC price?' }
  });
  
  fireEvent.click(screen.getByRole('button', { name: /submit/i }));
  
  await waitFor(() => {
    expect(screen.getByText(/bitcoin is trading/i)).toBeInTheDocument();
  });
});
```

### 3. Performance Optimization
```javascript
// Use React.memo for expensive components
const AssetCard = React.memo(({ asset }) => {
  return (
    <div className="asset-card">
      {/* Asset card content */}
    </div>
  );
});

// Debounce API calls
import { debounce } from 'lodash';

const debouncedQuery = debounce(async (query) => {
  const result = await api.askQuery(query);
  setResponse(result);
}, 500);
```

## 📱 **Mobile Considerations**

### Touch-Friendly Interface
```css
/* Ensure touch targets are at least 44px */
.touch-target {
  min-height: 44px;
  min-width: 44px;
  padding: 12px;
}

/* Optimize for thumb navigation */
.bottom-nav {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 60px;
}
```

### Progressive Web App Features
```javascript
// Add to manifest.json
{
  "name": "Pebble Crypto",
  "short_name": "PebbleCrypto",
  "description": "AI-powered cryptocurrency analysis",
  "start_url": "/",
  "display": "standalone",
  "theme_color": "#1f2937",
  "background_color": "#ffffff",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

## 🚀 **Deployment Checklist**

Before deploying your frontend:

- [ ] **API Integration Tested**: All endpoints working correctly
- [ ] **Error Handling**: Graceful error handling for all scenarios
- [ ] **Rate Limiting**: Proper rate limit handling and user feedback
- [ ] **Loading States**: Clear loading indicators for all async operations
- [ ] **Mobile Responsive**: Works well on all device sizes
- [ ] **Performance**: Optimized for fast loading and smooth interactions
- [ ] **Accessibility**: Proper ARIA labels and keyboard navigation
- [ ] **SEO**: Meta tags and structured data for search engines
- [ ] **Analytics**: User interaction tracking implemented
- [ ] **Security**: No sensitive data exposed in client-side code

---

**Last Updated:** January 2025  
**API Compatibility:** v1.0  
**Frontend Framework:** React (adaptable to Vue, Angular, etc.)  
**Test Coverage:** 100% API endpoint verification 