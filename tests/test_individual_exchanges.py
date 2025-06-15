#!/usr/bin/env python3
"""
Individual Exchange Test Script
Tests each exchange client individually to identify specific issues
"""

import asyncio
import sys
import os
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.services.binance import BinanceClient
from app.services.kucoin import KuCoinClient
from app.services.bybit import BybitClient
from app.services.gateio import GateIOClient
from app.services.bitget import BitgetClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExchangeTest")

async def test_exchange_client(exchange_name: str, client, symbol: str = "BTCUSDT"):
    """Test an individual exchange client"""
    print(f"\n🧪 Testing {exchange_name}...")
    
    try:
        # Test get_ticker method
        if hasattr(client, 'get_ticker'):
            ticker = await client.get_ticker(symbol)
            if ticker:
                print(f"✅ {exchange_name} ticker: ${ticker.get('price', 'N/A')}")
                return True
            else:
                print(f"❌ {exchange_name} returned None for ticker")
                return False
        else:
            print(f"❌ {exchange_name} doesn't have get_ticker method")
            return False
            
    except Exception as e:
        print(f"❌ {exchange_name} error: {str(e)}")
        return False

async def main():
    """Test all exchange clients individually"""
    print("🔍 INDIVIDUAL EXCHANGE CLIENT TESTING")
    print("=" * 50)
    
    exchanges = [
        ("Binance", BinanceClient()),
        ("KuCoin", KuCoinClient()),
        ("Bybit", BybitClient()),
        ("Gate.io", GateIOClient()),
        ("Bitget", BitgetClient())
    ]
    
    results = {}
    
    for name, client in exchanges:
        results[name] = await test_exchange_client(name, client)
    
    print("\n" + "=" * 50)
    print("📊 INDIVIDUAL TEST RESULTS:")
    print("=" * 50)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:12} {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    print(f"\nSuccess Rate: {success_count}/{total_count} ({success_rate:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main()) 