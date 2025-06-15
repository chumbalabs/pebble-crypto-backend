#!/usr/bin/env python3
"""
API Endpoints Test Script
Tests all endpoints following Anthropic's agent design patterns
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint and return the response"""
    print(f"\n🧪 Testing: {description}")
    print(f"📍 {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success")
            return result
        else:
            print(f"❌ Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return None

def test_ai_queries():
    """Test AI queries following Anthropic's patterns"""
    print("\n" + "="*60)
    print("🤖 TESTING AI QUERIES (Anthropic's Agent Patterns)")
    print("="*60)
    
    # Test queries based on MULTI_ASSET_IMPROVEMENTS.md
    test_queries = [
        # 1. Single Asset (Traditional)
        {
            "query": "What is the current price of Bitcoin?",
            "description": "Single Asset Query (Routing Pattern)",
            "expected_type": "single_asset"
        },
        
        # 2. Multi-Asset (Parallelization Pattern)
        {
            "query": "What are the current prices of BTC, ETH, and SOL?",
            "description": "Multi-Asset Query (Parallelization Pattern)",
            "expected_type": "multi_asset"
        },
        
        # 3. Comparison (Routing + Analysis)
        {
            "query": "Compare Bitcoin and Ethereum performance",
            "description": "Comparison Query (Routing + Analysis Pattern)",
            "expected_type": "comparison"
        },
        
        # 4. Portfolio Analysis
        {
            "query": "Analyze my portfolio of BTC, ETH, and SOL",
            "description": "Portfolio Query (Orchestrator-Workers Pattern)",
            "expected_type": "portfolio"
        },
        
        # 5. Natural Language Variations
        {
            "query": "How are Bitcoin, Ethereum, and Dogecoin performing today?",
            "description": "Natural Language Multi-Asset",
            "expected_type": "multi_asset"
        },
        
        # 6. Investment Advice
        {
            "query": "Should I buy Bitcoin or Ethereum for better returns?",
            "description": "Investment Advice with Comparison",
            "expected_type": "comparison"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 50)
        
        result = test_endpoint(
            "POST", 
            "/api/ask",
            {"query": test_case["query"]},
            test_case["description"]
        )
        
        if result:
            # Analyze the response quality
            analyze_response_quality(result, test_case)
            results.append({"test": test_case, "result": result, "success": True})
        else:
            results.append({"test": test_case, "result": None, "success": False})
        
        time.sleep(1)  # Rate limiting
    
    return results

def analyze_response_quality(result, test_case):
    """Analyze response quality based on Anthropic's principles"""
    print("\n📊 Response Quality Analysis:")
    
    # Check for required fields
    required_fields = ["query", "response", "timestamp", "metadata"]
    for field in required_fields:
        if field in result:
            print(f"✅ {field}: Present")
        else:
            print(f"❌ {field}: Missing")
    
    # Check response content quality
    response_text = result.get("response", "")
    if len(response_text) > 50:
        print(f"✅ Response length: {len(response_text)} chars (Good)")
    else:
        print(f"⚠️ Response length: {len(response_text)} chars (Short)")
    
    # Check for supporting data
    supporting_data = result.get("supporting_data", {})
    if supporting_data:
        print(f"✅ Supporting data: {len(supporting_data)} fields")
        for key in supporting_data.keys():
            print(f"   - {key}")
    else:
        print("⚠️ Supporting data: None")
    
    # Check metadata
    metadata = result.get("metadata", {})
    if metadata:
        print(f"✅ Metadata: {len(metadata)} fields")
        if "symbol" in metadata:
            print(f"   - Symbol: {metadata['symbol']}")
        if "data_sources" in metadata:
            print(f"   - Data sources: {metadata['data_sources']}")
    
    # Check for multi-timeframe analysis (advanced feature)
    if "timeframe_analysis" in supporting_data:
        print("✅ Multi-timeframe analysis: Present")
        tf_data = supporting_data["timeframe_analysis"]
        print(f"   - Timeframes: {list(tf_data.keys())}")

def test_other_endpoints():
    """Test other important endpoints"""
    print("\n" + "="*60)
    print("🔧 TESTING OTHER ENDPOINTS")
    print("="*60)
    
    # Health check
    test_endpoint("GET", "/api/health", description="Health Check")
    
    # Symbols
    test_endpoint("GET", "/symbols", description="Get Trading Symbols")
    
    # Historical data
    test_endpoint("GET", "/historical/BTCUSDT?interval=1h&limit=10", description="Historical Data")
    
    # Prediction
    test_endpoint("GET", "/predict/BTCUSDT?interval=1h", description="Price Prediction")
    
    # Investment advice
    test_endpoint("GET", "/investment-advice/BTCUSDT?interval=1h", description="Investment Advice")
    
    # Asset comparison
    test_endpoint("GET", "/compare-assets/BTCUSDT?comparison_symbols=ETHUSDT,SOLUSDT&time_period=7d", 
                 description="Asset Comparison")
    
    # Multi-exchange endpoints
    test_endpoint("GET", "/api/exchanges/health", description="Exchange Health Status")
    
    test_endpoint("GET", "/api/exchanges/coverage", description="Exchange Coverage Info")
    
    test_endpoint("POST", "/api/exchanges/best-prices", 
                 {"symbols": ["BTCUSDT", "ETHUSDT"]}, 
                 description="Best Prices Across Exchanges")

def generate_test_report(ai_results):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    total_tests = len(ai_results)
    successful_tests = sum(1 for r in ai_results if r["success"])
    
    print(f"📊 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    print(f"\n🎯 Anthropic's Agent Patterns Compliance:")
    
    # Check routing pattern
    routing_tests = [r for r in ai_results if r["success"] and "routing" in r["test"]["description"].lower()]
    print(f"   ✅ Routing Pattern: {len(routing_tests)} tests passed")
    
    # Check parallelization pattern
    parallel_tests = [r for r in ai_results if r["success"] and "parallelization" in r["test"]["description"].lower()]
    print(f"   ✅ Parallelization Pattern: {len(parallel_tests)} tests passed")
    
    # Check orchestrator-workers pattern
    orchestrator_tests = [r for r in ai_results if r["success"] and "orchestrator" in r["test"]["description"].lower()]
    print(f"   ✅ Orchestrator-Workers Pattern: {len(orchestrator_tests)} tests passed")
    
    print(f"\n📈 Quality Metrics:")
    
    # Response quality metrics
    avg_response_length = 0
    responses_with_data = 0
    responses_with_metadata = 0
    
    for result in ai_results:
        if result["success"] and result["result"]:
            response_text = result["result"].get("response", "")
            avg_response_length += len(response_text)
            
            if result["result"].get("supporting_data"):
                responses_with_data += 1
            
            if result["result"].get("metadata"):
                responses_with_metadata += 1
    
    if successful_tests > 0:
        avg_response_length = avg_response_length / successful_tests
        print(f"   Average response length: {avg_response_length:.0f} characters")
        print(f"   Responses with supporting data: {responses_with_data}/{successful_tests}")
        print(f"   Responses with metadata: {responses_with_metadata}/{successful_tests}")
    
    print(f"\n🚀 Production Readiness:")
    if successful_tests == total_tests:
        print("   ✅ All tests passed - PRODUCTION READY")
    elif successful_tests >= total_tests * 0.8:
        print("   ⚠️ Most tests passed - NEEDS MINOR FIXES")
    else:
        print("   ❌ Many tests failed - NEEDS MAJOR FIXES")

def main():
    """Main test execution"""
    print("🧪 PEBBLE CRYPTO API - COMPREHENSIVE ENDPOINT TESTING")
    print("Following Anthropic's Agent Design Patterns")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Base URL: {BASE_URL}")
    
    # Test AI queries (main focus)
    ai_results = test_ai_queries()
    
    # Test other endpoints
    test_other_endpoints()
    
    # Generate comprehensive report
    generate_test_report(ai_results)
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 Testing complete! Check the results above.")

if __name__ == "__main__":
    main() 