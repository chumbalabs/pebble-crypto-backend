#!/usr/bin/env python3
"""
Client-Side Data Quality Test
Tests actual data outputs from the API endpoints to verify quality and usefulness
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n📊 {title}")
    print('-'*40)

def format_json(data: Any, max_depth: int = 3, current_depth: int = 0) -> str:
    """Format JSON data with controlled depth for readability"""
    if current_depth >= max_depth:
        if isinstance(data, dict):
            return f"{{...{len(data)} fields...}}"
        elif isinstance(data, list):
            return f"[...{len(data)} items...]"
        else:
            return str(data)
    
    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            formatted_v = format_json(v, max_depth, current_depth + 1)
            items.append(f"  {'  ' * current_depth}{k}: {formatted_v}")
        return "{\n" + ",\n".join(items) + "\n" + "  " * current_depth + "}"
    elif isinstance(data, list):
        if len(data) == 0:
            return "[]"
        items = []
        for i, item in enumerate(data[:3]):  # Show first 3 items
            formatted_item = format_json(item, max_depth, current_depth + 1)
            items.append(f"  {'  ' * current_depth}[{i}]: {formatted_item}")
        if len(data) > 3:
            items.append(f"  {'  ' * current_depth}... {len(data) - 3} more items")
        return "[\n" + ",\n".join(items) + "\n" + "  " * current_depth + "]"
    else:
        return json.dumps(data) if isinstance(data, str) else str(data)

def test_ai_query_outputs():
    """Test AI query endpoints and examine actual outputs"""
    print_section("AI QUERY DATA OUTPUTS")
    
    test_queries = [
        {
            "query": "What is the current price of Bitcoin?",
            "description": "Single Asset Price Query",
            "expected_data": ["current_price", "symbol", "timestamp"]
        },
        {
            "query": "Compare Bitcoin and Ethereum performance",
            "description": "Multi-Asset Comparison",
            "expected_data": ["comparison_metrics", "performance_ranking"]
        },
        {
            "query": "Analyze my portfolio of BTC, ETH, and SOL",
            "description": "Portfolio Analysis",
            "expected_data": ["portfolio_metrics", "diversification"]
        },
        {
            "query": "Should I buy Bitcoin now based on technical indicators?",
            "description": "Investment Advice with Technical Analysis",
            "expected_data": ["technical_indicators", "trading_recommendations"]
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print_subsection(f"{i}. {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/ask",
                json={"query": test_case["query"]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Status: {response.status_code}")
                print(f"📝 Response: {data.get('response', 'No response')}")
                
                # Examine supporting data
                supporting_data = data.get('supporting_data', {})
                if supporting_data:
                    print(f"\n📊 Supporting Data ({len(supporting_data)} fields):")
                    for key, value in supporting_data.items():
                        print(f"  • {key}: {type(value).__name__}")
                        if isinstance(value, dict) and len(value) <= 5:
                            for sub_key, sub_value in value.items():
                                print(f"    - {sub_key}: {sub_value}")
                        elif isinstance(value, list) and len(value) <= 3:
                            for idx, item in enumerate(value):
                                print(f"    [{idx}]: {item}")
                
                # Examine metadata
                metadata = data.get('metadata', {})
                if metadata:
                    print(f"\n🏷️ Metadata:")
                    for key, value in metadata.items():
                        print(f"  • {key}: {value}")
                
                # Check for multi-timeframe data
                if 'timeframe_analysis' in supporting_data:
                    print(f"\n⏰ Multi-Timeframe Analysis:")
                    tf_data = supporting_data['timeframe_analysis']
                    for timeframe, tf_info in tf_data.items():
                        print(f"  • {timeframe}: {tf_info}")
                
                print(f"\n📏 Response Quality:")
                print(f"  • Response length: {len(data.get('response', ''))} characters")
                print(f"  • Has supporting data: {'✅' if supporting_data else '❌'}")
                print(f"  • Has metadata: {'✅' if metadata else '❌'}")
                print(f"  • Data sources: {metadata.get('data_sources', [])}")
                
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        time.sleep(1)  # Rate limiting

def test_market_data_outputs():
    """Test market data endpoints and examine outputs"""
    print_section("MARKET DATA OUTPUTS")
    
    endpoints = [
        {
            "url": "/symbols?sort_by=volume&descending=true",
            "method": "GET",
            "description": "Trading Symbols (Volume Sorted)",
            "expected_fields": ["symbols", "sorting", "timestamp"]
        },
        {
            "url": "/historical/BTCUSDT?interval=1h&limit=5",
            "method": "GET", 
            "description": "Historical OHLCV Data",
            "expected_fields": ["historical_data", "symbol", "interval"]
        },
        {
            "url": "/predict/BTCUSDT?interval=1h",
            "method": "GET",
            "description": "Price Prediction with Technical Analysis",
            "expected_fields": ["price_analysis", "ai_insights", "metadata"]
        },
        {
            "url": "/investment-advice/BTCUSDT?interval=1h",
            "method": "GET",
            "description": "Investment Advice",
            "expected_fields": ["advice", "symbol", "interval"]
        }
    ]
    
    for i, endpoint in enumerate(endpoints, 1):
        print_subsection(f"{i}. {endpoint['description']}")
        print(f"Endpoint: {endpoint['method']} {endpoint['url']}")
        
        try:
            response = requests.get(f"{BASE_URL}{endpoint['url']}", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                
                # Show data structure
                print(f"\n📊 Data Structure:")
                print(format_json(data, max_depth=2))
                
                # Check expected fields
                print(f"\n✅ Field Validation:")
                for field in endpoint['expected_fields']:
                    if field in data:
                        value = data[field]
                        print(f"  • {field}: ✅ ({type(value).__name__})")
                        if isinstance(value, list):
                            print(f"    - Length: {len(value)} items")
                        elif isinstance(value, dict):
                            print(f"    - Keys: {list(value.keys())[:5]}")
                    else:
                        print(f"  • {field}: ❌ Missing")
                
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        time.sleep(1)

def test_multi_exchange_outputs():
    """Test multi-exchange endpoints and examine outputs"""
    print_section("MULTI-EXCHANGE DATA OUTPUTS")
    
    endpoints = [
        {
            "url": "/api/exchanges/health",
            "method": "GET",
            "description": "Exchange Health Status",
            "data": None
        },
        {
            "url": "/api/exchanges/coverage", 
            "method": "GET",
            "description": "Exchange Coverage Information",
            "data": None
        },
        {
            "url": "/api/exchanges/best-prices",
            "method": "POST",
            "description": "Best Prices Across Exchanges",
            "data": {"symbols": ["BTCUSDT", "ETHUSDT"]}
        }
    ]
    
    for i, endpoint in enumerate(endpoints, 1):
        print_subsection(f"{i}. {endpoint['description']}")
        print(f"Endpoint: {endpoint['method']} {endpoint['url']}")
        
        try:
            if endpoint['method'] == 'GET':
                response = requests.get(f"{BASE_URL}{endpoint['url']}", timeout=15)
            else:
                response = requests.post(
                    f"{BASE_URL}{endpoint['url']}", 
                    json=endpoint['data'],
                    timeout=15
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                
                # Show key metrics
                if 'exchanges' in data:
                    exchanges = data['exchanges']
                    print(f"\n🏢 Exchange Information:")
                    for exchange_name, exchange_info in exchanges.items():
                        if isinstance(exchange_info, dict):
                            status = exchange_info.get('status', 'unknown')
                            print(f"  • {exchange_name}: {status}")
                            if 'response_time' in exchange_info:
                                print(f"    - Response time: {exchange_info['response_time']}ms")
                
                if 'total_exchanges' in data:
                    print(f"\n📊 Summary:")
                    print(f"  • Total exchanges: {data.get('total_exchanges', 0)}")
                    print(f"  • Healthy exchanges: {data.get('healthy_exchanges', 0)}")
                
                # Show data structure (limited depth)
                print(f"\n📊 Data Structure Preview:")
                print(format_json(data, max_depth=2))
                
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        time.sleep(1)

def test_real_world_scenarios():
    """Test real-world usage scenarios"""
    print_section("REAL-WORLD USAGE SCENARIOS")
    
    scenarios = [
        {
            "name": "Day Trader Scenario",
            "query": "What's the volatility of Bitcoin in the last hour and should I enter a position?",
            "expected_insights": ["volatility", "entry_points", "risk_assessment"]
        },
        {
            "name": "Portfolio Manager Scenario", 
            "query": "How are my top 3 holdings BTC, ETH, and SOL performing compared to each other?",
            "expected_insights": ["relative_performance", "correlation", "diversification"]
        },
        {
            "name": "Arbitrage Trader Scenario",
            "description": "Check best prices across exchanges",
            "endpoint": "/api/exchanges/best-prices",
            "data": {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
            "expected_insights": ["price_differences", "arbitrage_opportunities"]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print_subsection(f"{i}. {scenario['name']}")
        
        if 'query' in scenario:
            # AI query scenario
            print(f"Query: '{scenario['query']}'")
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/ask",
                    json={"query": scenario["query"]},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Response received")
                    print(f"📝 AI Response: {data.get('response', '')}")
                    
                    # Check for expected insights
                    supporting_data = data.get('supporting_data', {})
                    print(f"\n🎯 Expected Insights Check:")
                    for insight in scenario['expected_insights']:
                        found = any(insight.lower() in str(v).lower() 
                                  for v in supporting_data.values())
                        print(f"  • {insight}: {'✅' if found else '❌'}")
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                
        elif 'endpoint' in scenario:
            # Direct endpoint scenario
            print(f"Endpoint: {scenario['endpoint']}")
            
            try:
                response = requests.post(
                    f"{BASE_URL}{scenario['endpoint']}",
                    json=scenario['data'],
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Data received")
                    print(f"📊 Results: {format_json(data, max_depth=2)}")
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
        
        time.sleep(2)

def main():
    """Run comprehensive client-side data quality tests"""
    print("🧪 PEBBLE CRYPTO API - CLIENT-SIDE DATA QUALITY TEST")
    print("Testing actual data outputs and usefulness from client perspective")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test different categories
    test_ai_query_outputs()
    test_market_data_outputs() 
    test_multi_exchange_outputs()
    test_real_world_scenarios()
    
    print_section("SUMMARY")
    print("✅ Client-side data quality testing completed!")
    print("📊 Check the outputs above to verify:")
    print("  • Data completeness and structure")
    print("  • Response quality and usefulness") 
    print("  • Real-world scenario applicability")
    print("  • Multi-exchange integration effectiveness")
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 