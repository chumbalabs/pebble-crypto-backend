# Pebble Crypto Frontend Development Guide

## Overview
This guide outlines the architecture, implementation, and best practices for developing the Pebble Crypto frontend application. The frontend will provide an intuitive, responsive interface for users to access the powerful cryptocurrency analytics capabilities offered by the backend API.

## Architecture & Tech Stack

### Recommended Technology Stack
- **Framework**: React.js with TypeScript
- **State Management**: Redux Toolkit for global state, React Query for API data
- **Styling**: Tailwind CSS with custom theme
- **Charts**: Recharts for simple, elegant line charts
- **Dashboard Components**: Tremor for analytics components
- **Additional Libraries**:
  - date-fns - Date manipulation
  - react-router-dom - Routing
  - axios - API requests
  - react-hook-form - Form handling
  - zod - Form validation
  - framer-motion - Animations
  - vitest - Testing

### Project Structure
```
src/
├── assets/               # Static assets, icons, etc.
├── components/           # Reusable UI components
│   ├── charts/           # Chart components
│   ├── dashboard/        # Dashboard widgets  
│   ├── layout/           # Layout components
│   └── ui/               # Basic UI components
├── features/             # Feature-based components
│   ├── market-analysis/  # Market analysis feature
│   ├── predictions/      # Prediction features
│   └── ai-assistant/     # AI assistant chat interface
├── hooks/                # Custom React hooks
├── services/             # API services
├── store/                # Redux store configuration
├── types/                # TypeScript type definitions
└── utils/                # Utility functions
```

## API Integration

### Core Endpoints

#### 1. AI Agent Query
```typescript
// API Service Example
export const askAgent = async (question: string): Promise<AgentResponse> => {
  try {
    const response = await axios.post('/api/ask', { question });
    return response.data;
  } catch (error) {
    console.error('Error querying AI agent:', error);
    throw error;
  }
};
```

**Response Structure**:
```typescript
interface AgentResponse {
  query: string;
  response: string;
  timestamp: string;
  supporting_data: {
    current_price?: number;
    price_change_24h?: number;
    rsi?: number;
    bollinger_signal?: string;
    percent_b?: number;
    prediction?: any;
    market_summary?: string;
    observations?: string[];
  };
  metadata: {
    symbol: string;
    interval: string;
    data_sources: string[];
  };
  multi_timeframe?: Record<string, TimeframeData>;
}

interface TimeframeData {
  current_price: number;
  price_change: number;
  high: number;
  low: number;
  volume_avg: number;
  volatility: number;
  indicators?: {
    bollinger_bands: any;
    rsi: number;
    sma_20: number;
    sma_50: number;
    sma_20_diff: number;
    sma_50_diff: number;
  };
  trend?: {
    direction: string;
    strength: number;
    description: string;
  };
}
```

#### 2. Market Data
```typescript
export const getMarketData = async (
  symbol: string,
  interval: string = '1h',
  limit: number = 100
): Promise<MarketData> => {
  try {
    const response = await axios.get('/api/market/data', {
      params: { symbol, interval, limit }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching market data:', error);
    throw error;
  }
};
```

#### 3. Technical Analysis
```typescript
export const getTechnicalAnalysis = async (
  symbol: string,
  interval: string = '1h'
): Promise<TechnicalAnalysis> => {
  try {
    const response = await axios.get('/api/technical/analysis', {
      params: { symbol, interval }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching technical analysis:', error);
    throw error;
  }
};
```

## Core Features Implementation

### 1. AI Assistant Chat Interface

Create a conversational UI for interacting with the AI agent. Use a chat-like interface where users can ask questions about cryptocurrencies and receive detailed responses.

```jsx
const AIChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    const userMessage = {
      id: Date.now(),
      content: inputValue,
      sender: 'user'
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      const response = await askAgent(inputValue);
      
      const botMessage = {
        id: Date.now() + 1,
        content: response.response,
        sender: 'bot',
        data: response
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      // Handle error
      const errorMessage = {
        id: Date.now() + 1,
        content: 'Sorry, I encountered an error processing your request.',
        sender: 'bot',
        error: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {isLoading && <TypingIndicator />}
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about any cryptocurrency..."
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2"
          />
          <button 
            type="submit" 
            disabled={isLoading}
            className="bg-beige-600 text-white px-4 py-2 rounded-lg hover:bg-beige-700 transition"
          >
            {isLoading ? 'Thinking...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
};
```

### 2. Multi-Timeframe Analysis Dashboard

Create a dashboard that displays the multi-timeframe analysis with tabs for different timeframes (1h, 4h, 1d, 1w).

```jsx
const MultiTimeframeDashboard = ({ symbol }) => {
  const { data, isLoading, error } = useQuery(
    ['multi-timeframe', symbol],
    () => getMarketData(symbol)
  );
  
  const [activeTimeframe, setActiveTimeframe] = useState('1h');
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay error={error} />;
  
  const timeframes = Object.keys(data.multi_timeframe || {});
  const activeData = data?.multi_timeframe?.[activeTimeframe];
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">{symbol} Analysis</h2>
        <div className="flex space-x-2">
          {timeframes.map(tf => (
            <button
              key={tf}
              onClick={() => setActiveTimeframe(tf)}
              className={`px-3 py-1 rounded ${
                activeTimeframe === tf 
                  ? 'bg-beige-600 text-white' 
                  : 'bg-gray-200'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>
      
      {activeData && (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          <MetricCard 
            title="Current Price" 
            value={`$${activeData.current_price.toFixed(6)}`} 
          />
          <MetricCard 
            title="Price Change" 
            value={`${(activeData.price_change * 100).toFixed(2)}%`}
            trend={activeData.price_change > 0 ? 'up' : 'down'} 
          />
          <MetricCard 
            title="Volatility" 
            value={`${activeData.volatility.toFixed(2)}%`} 
          />
          <MetricCard 
            title="RSI" 
            value={activeData.indicators?.rsi.toFixed(2) || 'N/A'} 
          />
          <MetricCard 
            title="Trend" 
            value={activeData.trend?.description || 'N/A'} 
          />
          <MetricCard 
            title="Volume Avg" 
            value={`${formatNumber(activeData.volume_avg)}`} 
          />
        </div>
      )}
      
      {activeData && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Technical Indicators</h3>
          <TechnicalIndicatorsTable indicators={activeData.indicators} />
        </div>
      )}
    </div>
  );
};
```

### 3. Simple Line Chart Component

Implement a clean, minimalist line chart for price data visualization.

```jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useState, useEffect } from 'react';

const PriceChart = ({ symbol, interval, data }) => {
  const [chartData, setChartData] = useState([]);
  
  useEffect(() => {
    if (!data || data.length === 0) return;
    
    // Format data for the chart
    const formattedData = data.map(item => ({
      timestamp: new Date(item.timestamp).toLocaleDateString(),
      price: item.close,
      volume: item.volume,
    }));
    
    setChartData(formattedData);
  }, [data]);
  
  return (
    <div className="w-full bg-white rounded-lg shadow-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">{symbol} Price Chart ({interval})</h2>
      </div>
      
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
            <XAxis 
              dataKey="timestamp" 
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => {
                // Format the date to keep the chart clean
                return value;
              }}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              domain={['auto', 'auto']}
              tickFormatter={(value) => `$${value.toFixed(2)}`}
            />
            <Tooltip 
              formatter={(value) => [`$${value.toFixed(6)}`, 'Price']}
              labelFormatter={(label) => `Date: ${label}`}
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                padding: '10px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#D2B48C" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, fill: '#D2B48C' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {/* Optional volume chart */}
      <div className="h-32 w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
            <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
            <YAxis 
              tick={{ fontSize: 10 }}
              tickFormatter={(value) => formatVolume(value)}
            />
            <Tooltip 
              formatter={(value) => [formatVolume(value), 'Volume']}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Line 
              type="monotone" 
              dataKey="volume" 
              stroke="#A89B89" 
              strokeWidth={1}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Utility function to format volume
const formatVolume = (value) => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(2)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)}K`;
  return value.toString();
};

export default PriceChart;
```

### 4. Market Analysis and Prediction Interface

Create a detailed market analysis page with predictions.

```jsx
const MarketAnalysis = ({ symbol }) => {
  const { data, isLoading, error } = useQuery(
    ['market-analysis', symbol],
    () => getTechnicalAnalysis(symbol)
  );
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay error={error} />;
  
  const { prediction, price_analysis, key_levels, confidence_score } = data;
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-6">{symbol} Market Analysis</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-xl font-semibold mb-4">Price Analysis</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-beige-50 rounded-lg">
              <span className="font-medium">Current Price</span>
              <span className="text-lg">${price_analysis.current.toFixed(6)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-beige-50 rounded-lg">
              <span className="font-medium">Predicted Price</span>
              <span className="text-lg">${prediction.next_period.toFixed(6)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-beige-50 rounded-lg">
              <span className="font-medium">Confidence Score</span>
              <span className="text-lg">{(confidence_score * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-beige-50 rounded-lg">
              <span className="font-medium">Prediction Range</span>
              <span className="text-lg">
                ${prediction.range.low.toFixed(6)} - ${prediction.range.high.toFixed(6)}
              </span>
            </div>
          </div>
        </div>
        
        <div>
          <h3 className="text-xl font-semibold mb-4">Key Levels</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
              <span className="font-medium">Support</span>
              <span className="text-lg">${key_levels.support.toFixed(6)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
              <span className="font-medium">Resistance</span>
              <span className="text-lg">${key_levels.resistance.toFixed(6)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-beige-50 rounded-lg">
              <span className="font-medium">Trend Strength</span>
              <span className="text-lg">{key_levels.trend_strength.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-4">Technical Indicators</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <IndicatorCard
            title="RSI"
            value={data.indicators.rsi}
            status={
              data.indicators.rsi > 70 ? 'Overbought' :
              data.indicators.rsi < 30 ? 'Oversold' : 'Neutral'
            }
            statusColor={
              data.indicators.rsi > 70 ? 'red' :
              data.indicators.rsi < 30 ? 'green' : 'beige'
            }
          />
          <IndicatorCard
            title="Bollinger Bands"
            value={data.indicators.bollinger_bands.signal}
            status={data.indicators.bollinger_bands.signal}
            statusColor={
              data.indicators.bollinger_bands.signal === 'BUY' ? 'green' :
              data.indicators.bollinger_bands.signal === 'SELL' ? 'red' : 'beige'
            }
          />
          <IndicatorCard
            title="MACD"
            value={data.indicators.macd.histogram[data.indicators.macd.histogram.length - 1].toFixed(6)}
            status={
              data.indicators.macd.histogram[data.indicators.macd.histogram.length - 1] > 0 ? 'Bullish' : 'Bearish'
            }
            statusColor={
              data.indicators.macd.histogram[data.indicators.macd.histogram.length - 1] > 0 ? 'green' : 'red'
            }
          />
        </div>
      </div>
      
      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-4">AI Insights</h3>
        <div className="p-4 bg-beige-50 rounded-lg">
          <p className="text-lg mb-4">{data.ai_insights.market_summary}</p>
          <h4 className="font-medium mb-2">Key Observations:</h4>
          <ul className="list-disc pl-6 space-y-2">
            {data.ai_insights.technical_observations.map((observation, index) => (
              <li key={index}>{observation}</li>
            ))}
          </ul>
          <h4 className="font-medium mt-4 mb-2">Recommendations:</h4>
          <ul className="list-disc pl-6 space-y-2">
            {data.ai_insights.trading_recommendations.map((recommendation, index) => (
              <li key={index}>{recommendation}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
```

## UI/UX Guidelines

### 1. Design Principles
- **Clean & Professional**: Use a clean, professional design that instills confidence in the data
- **Actionable Information**: Prioritize actionable information and clear insights
- **Visual Hierarchy**: Create a clear visual hierarchy to direct users to the most important information
- **Color Psychology**: Use colors meaningfully to indicate market trends (green/red) and data significance
- **Progressive Disclosure**: Organize information with progressive disclosure, showing essential data first

### 2. Color Palette
- **Primary**: #D2B48C (Beige/Tan - brand color)
- **Secondary**: #A89B89 (Taupe - neutral elements)
- **Success**: #10b981 (Green - positive trends)
- **Danger**: #ef4444 (Red - negative trends)
- **Warning**: #f59e0b (Yellow - alerts)
- **Info**: #8BC9D0 (Light blue - informational)
- **Background**: #F8F6F2 (Light beige - main background)
- **Card Background**: #ffffff (White - component background)
- **Text Primary**: #3D3C38 (Dark brown - main text)
- **Text Secondary**: #847E70 (Medium brown - secondary text)

### 3. Typography
- **Primary Font**: Inter (sans-serif)
- **Headings**: Inter Semi-Bold (600)
- **Body**: Inter Regular (400)
- **Mono**: JetBrains Mono (for code and numerical data)

### 4. Responsive Design
- Desktop-first approach with responsive breakpoints for tablet and mobile
- Custom mobile views for complex charts and data tables
- Touch-friendly controls for mobile users
- Consider using simplified views on smaller screens

## Core Components

### 1. Header
```jsx
const Header = () => {
  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <img src="/logo.svg" alt="Pebble Crypto" className="h-8 w-auto" />
            <h1 className="ml-3 text-xl font-semibold text-gray-900">Pebble Crypto</h1>
          </div>
          <nav className="flex space-x-8">
            <NavLink to="/" activeClassName="text-beige-600" className="text-gray-500 hover:text-gray-900">
              Dashboard
            </NavLink>
            <NavLink to="/analysis" activeClassName="text-beige-600" className="text-gray-500 hover:text-gray-900">
              Analysis
            </NavLink>
            <NavLink to="/assistant" activeClassName="text-beige-600" className="text-gray-500 hover:text-gray-900">
              AI Assistant
            </NavLink>
          </nav>
          <div>
            <CoinSelector />
          </div>
        </div>
      </div>
    </header>
  );
};
```

### 2. Coin Selector
```jsx
const CoinSelector = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const { data: coins } = useQuery('available-coins', fetchAvailableCoins);
  const { selectedCoin, setSelectedCoin } = useSelectedCoin();
  
  const filteredCoins = coins?.filter(coin => 
    coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];
  
  return (
    <div className="relative">
      <button 
        className="flex items-center bg-beige-100 px-4 py-2 rounded-lg hover:bg-beige-200 transition"
        onClick={() => setIsOpen(!isOpen)}
      >
        {selectedCoin ? (
          <>
            <img src={selectedCoin.icon} alt={selectedCoin.symbol} className="h-5 w-5 mr-2" />
            <span>{selectedCoin.symbol}</span>
          </>
        ) : (
          <span>Select Coin</span>
        )}
        <ChevronDownIcon className="h-5 w-5 ml-2" />
      </button>
      
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-xl z-10">
          <div className="p-2">
            <input
              type="text"
              placeholder="Search coins..."
              className="w-full p-2 border rounded"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
          </div>
          <ul className="max-h-96 overflow-y-auto">
            {filteredCoins.map(coin => (
              <li key={coin.symbol}>
                <button
                  className="w-full flex items-center px-4 py-2 hover:bg-beige-50 transition"
                  onClick={() => {
                    setSelectedCoin(coin);
                    setIsOpen(false);
                  }}
                >
                  <img src={coin.icon} alt={coin.symbol} className="h-5 w-5 mr-2" />
                  <span>{coin.name}</span>
                  <span className="ml-2 text-gray-400">{coin.symbol}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
```

### 3. Dashboard Page
```jsx
const Dashboard = () => {
  const { selectedCoin } = useSelectedCoin();
  const [timeframe, setTimeframe] = useState('1h');
  
  if (!selectedCoin) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <h2 className="text-xl font-medium mb-4">Welcome to Pebble Crypto</h2>
        <p className="text-gray-500 mb-6">Please select a coin to view analytics</p>
        <CoinSelector />
      </div>
    );
  }
  
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">{selectedCoin.name} Dashboard</h1>
        <p className="text-gray-500">Comprehensive analytics and insights</p>
      </div>
      
      <div className="grid grid-cols-1 gap-8">
        <PriceChart symbol={selectedCoin.symbol} interval={timeframe} />
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <MultiTimeframeDashboard symbol={selectedCoin.symbol} />
          </div>
          <div>
            <PriceCard symbol={selectedCoin.symbol} />
            <div className="mt-6">
              <KeyLevelsCard symbol={selectedCoin.symbol} />
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TechnicalIndicatorsCard symbol={selectedCoin.symbol} />
          <MarketInsightsCard symbol={selectedCoin.symbol} />
        </div>
      </div>
    </div>
  );
};
```

## State Management

### 1. Setting up Redux Store

```jsx
// store/index.js
import { configureStore } from '@reduxjs/toolkit';
import marketDataReducer from './marketDataSlice';
import uiReducer from './uiSlice';
import settingsReducer from './settingsSlice';

export const store = configureStore({
  reducer: {
    marketData: marketDataReducer,
    ui: uiReducer,
    settings: settingsReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### 2. Setting up React Query

```jsx
// App.tsx
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 30, // 30 minutes
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/assistant" element={<Assistant />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
```

## Performance Optimization

1. **Lazy Loading & Code Splitting**
   - Use React.lazy() and Suspense for component-level code-splitting
   - Use dynamic imports for large modules and libraries

2. **Memoization**
   - Use useMemo for expensive calculations
   - Use React.memo for pure components that render often

3. **Virtual List Rendering**
   - Use virtualized lists for long data sets (e.g., market data table)

4. **Web Workers**
   - Offload heavy calculations to web workers
   - Consider using Comlink for easier web worker communication

5. **Cached APIs with React Query**
   - Implement appropriate caching strategies
   - Use background refetching for fresh data

## Accessibility Guidelines

1. **Semantic HTML**
   - Use proper heading hierarchy
   - Use semantic elements (nav, main, section, etc.)

2. **Keyboard Navigation**
   - Ensure all interactive elements are keyboard accessible
   - Implement focus states and tab order

3. **Screen Reader Support**
   - Add aria attributes where necessary
   - Test with screen readers

4. **Color Contrast**
   - Ensure text meets WCAG AA standards for contrast
   - Don't rely solely on color for conveying information

## Deployment Recommendations

1. **Build Pipeline**
   - Use CI/CD pipelines for automated testing and deployment
   - Implement staged releases (dev, staging, production)

2. **Performance Monitoring**
   - Implement Real User Monitoring (RUM)
   - Track core web vitals

3. **Error Tracking**
   - Implement error boundaries in React
   - Use error tracking service

4. **Analytics**
   - Track user interactions to improve UX
   - Monitor feature usage

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`
4. Configure the API URL in the .env file

## Next Steps

- Implement user authentication
- Add portfolio tracking feature
- Create comparison views for multiple cryptocurrencies
- Implement alerts and notifications
- Add localization for multiple languages
