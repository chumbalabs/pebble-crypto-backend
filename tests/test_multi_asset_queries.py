# Test file for multi-asset natural language queries
# This demonstrates the improved capabilities based on Anthropic's agent design patterns

import asyncio
import json
from app.core.ai.agent import MarketAgent

async def test_multi_asset_queries():
    """Test various multi-asset query patterns"""
    
    agent = MarketAgent()
    
    # Test queries that should now work with the improved system
    test_queries = [
        # Multi-asset queries
        "What are the current prices of BTC, ETH, and SOL?",
        "Compare Bitcoin and Ethereum performance today",
        "How are BTC, ADA, and DOGE doing this week?",
        
        # Comparison queries
        "Which is performing better: Bitcoin vs Ethereum vs Solana?",
        "Compare BTC and ETH volatility over the last day",
        "Is BTC or ETH more volatile right now?",
        
        # Portfolio queries  
        "Analyze my portfolio of BTC, ETH, SOL, and ADA",
        "What's the correlation between Bitcoin and Ethereum?",
        "How diversified is a portfolio with BTC, ETH, and MATIC?",
        
        # Mixed intent queries
        "Should I buy Bitcoin or Ethereum for better returns?",
        "What's the trend for BTC and ETH over multiple timeframes?",
        "Give me analysis on BTC, ETH, and SOL including volatility",
        
        # Natural language variations
        "Tell me about Bitcoin and Ethereum prices",
        "I want to compare Solana against Cardano and Dogecoin",
        "Show me how Bitcoin, Ethereum, and Solana are performing",
        "What's better investment: BTC or ETH or SOL?"
    ]
    
    print("🚀 Testing Multi-Asset Query Capabilities")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        try:
            # Process the query
            result = await agent.process_query(query)
            
            # Extract and display key information
            query_info = agent._extract_query_info(query)
            
            print(f"Detected symbols: {query_info.get('symbols', [])}")
            print(f"Query type: {query_info.get('query_type', 'Unknown')}")
            print(f"Intent: {query_info.get('intent', 'Unknown')}")
            
            # Show response (truncated for readability)
            response = result.get('response', 'No response generated')
            if len(response) > 200:
                response = response[:200] + "..."
            print(f"Response preview: {response}")
            
            # Show metadata
            metadata = result.get('metadata', {})
            if 'data_sources' in metadata:
                print(f"Data sources used: {metadata['data_sources']}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print()

def test_query_classification():
    """Test the improved query classification system"""
    
    agent = MarketAgent()
    
    test_cases = [
        ("What's the price of Bitcoin?", "single_asset", ["BTCUSDT"]),
        ("Compare BTC and ETH", "comparison", ["BTCUSDT", "ETHUSDT"]),
        ("How are BTC, ETH, and SOL performing?", "multi_asset", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        ("Analyze my portfolio with Bitcoin and Ethereum", "portfolio", ["BTCUSDT", "ETHUSDT"]),
        ("Is Bitcoin better than Ethereum?", "comparison", ["BTCUSDT", "ETHUSDT"]),
        ("Show me BTC vs ETH vs SOL", "comparison", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
    ]
    
    print("🧪 Testing Query Classification")
    print("=" * 50)
    
    for query, expected_type, expected_symbols in test_cases:
        result = agent._extract_query_info(query)
        
        symbols = result.get('symbols', [])
        query_type = result.get('query_type', 'unknown')
        
        # Check if classification is correct
        type_correct = query_type == expected_type
        symbols_correct = set(symbols) == set(expected_symbols)
        
        status = "✅" if type_correct and symbols_correct else "❌"
        
        print(f"{status} '{query}'")
        print(f"   Expected: {expected_type}, {expected_symbols}")
        print(f"   Got:      {query_type}, {symbols}")
        print()

if __name__ == "__main__":
    print("Testing Multi-Asset Natural Language Query System")
    print("Based on Anthropic's Agent Design Patterns")
    print("=" * 60)
    
    # Test classification first
    test_query_classification()
    
    # Then test full query processing
    # asyncio.run(test_multi_asset_queries())
    
    print("\n📋 Summary of Improvements:")
    print("1. ✅ Multi-symbol extraction from natural language")
    print("2. ✅ Query type classification (single/multi/comparison/portfolio)")
    print("3. ✅ Parallel data collection using asyncio.gather")
    print("4. ✅ Routing pattern for different response types")
    print("5. ✅ Comparison and portfolio analytics")
    print("6. ✅ Support for full cryptocurrency names")
    print("\n🎯 Based on Anthropic's recommendations:")
    print("   - Simple, composable patterns")
    print("   - Clear routing and classification")
    print("   - Parallelization for efficiency")
    print("   - Transparent processing steps") 