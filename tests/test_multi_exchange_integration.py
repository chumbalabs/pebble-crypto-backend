#!/usr/bin/env python3
"""
Multi-Exchange Integration Test Script
Tests the new multi-exchange capabilities of the Pebble Crypto Backend
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.core.ai.agent import MarketAgent
from app.services.exchange_aggregator import ExchangeAggregator
from app.services.binance import BinanceClient
from app.services.kucoin import KuCoinClient
from app.services.bybit import BybitClient
from app.services.gateio import GateIOClient
from app.services.bitget import BitgetClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MultiExchangeTest")

class MultiExchangeIntegrationTest:
    """Test suite for multi-exchange integration"""
    
    def __init__(self):
        self.agent = None
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    async def setup(self):
        """Initialize the market agent and components"""
        try:
            logger.info("🚀 Initializing Multi-Exchange Integration Test")
            self.agent = MarketAgent()
            
            # Give the agent time to initialize
            await asyncio.sleep(2)
            
            logger.info("✅ Market agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize market agent: {str(e)}")
            self.results["errors"].append(f"Setup failed: {str(e)}")
            return False
    
    async def test_exchange_registration(self):
        """Test that all exchanges are properly registered"""
        logger.info("\n📋 Testing Exchange Registration...")
        
        try:
            # Check if all exchanges are registered
            expected_exchanges = ["binance", "kucoin", "bybit", "gateio", "bitget"]
            registered_exchanges = list(self.agent.exchange_aggregator.exchanges.keys())
            
            for exchange in expected_exchanges:
                if exchange in registered_exchanges:
                    logger.info(f"✅ {exchange.capitalize()} exchange registered")
                else:
                    logger.error(f"❌ {exchange.capitalize()} exchange NOT registered")
                    self.results["failed"] += 1
                    return False
            
            logger.info(f"✅ All {len(expected_exchanges)} exchanges registered successfully")
            self.results["passed"] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Exchange registration test failed: {str(e)}")
            self.results["errors"].append(f"Exchange registration: {str(e)}")
            self.results["failed"] += 1
            return False
    
    async def test_exchange_health(self):
        """Test exchange health monitoring"""
        logger.info("\n🏥 Testing Exchange Health Monitoring...")
        
        try:
            health_data = await self.agent.get_exchange_health()
            
            if health_data.get("status") == "success":
                exchanges = health_data.get("exchanges", {})
                total_exchanges = health_data.get("total_exchanges", 0)
                healthy_exchanges = health_data.get("healthy_exchanges", 0)
                
                logger.info(f"✅ Health check completed: {healthy_exchanges}/{total_exchanges} exchanges healthy")
                
                for exchange_name, health_info in exchanges.items():
                    status = health_info.get("status", "unknown")
                    response_time = health_info.get("response_time", "N/A")
                    
                    if status == "healthy":
                        logger.info(f"  ✅ {exchange_name}: {status} ({response_time:.2f}s)" if isinstance(response_time, float) else f"  ✅ {exchange_name}: {status}")
                    else:
                        logger.warning(f"  ⚠️ {exchange_name}: {status}")
                
                self.results["passed"] += 1
                return True
            else:
                logger.error(f"❌ Health check failed: {health_data.get('error', 'Unknown error')}")
                self.results["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Exchange health test failed: {str(e)}")
            self.results["errors"].append(f"Exchange health: {str(e)}")
            self.results["failed"] += 1
            return False
    
    async def test_single_asset_query(self):
        """Test single asset data collection with multi-exchange support"""
        logger.info("\n💰 Testing Single Asset Query with Multi-Exchange...")
        
        try:
            test_symbol = "BTCUSDT"
            query = f"What is the current price of {test_symbol}?"
            
            logger.info(f"Query: '{query}'")
            response = await self.agent.process_query(query)
            
            if response.get("status") == "success":
                data = response.get("data", {})
                answer = response.get("answer", "")
                
                # Check if we got exchange information
                exchange_used = data.get("exchange_used")
                current_price = data.get("current_price")
                arbitrage_data = data.get("arbitrage")
                
                logger.info(f"✅ Query processed successfully")
                logger.info(f"  📊 Exchange used: {exchange_used}")
                logger.info(f"  💵 Current price: ${current_price}")
                
                if arbitrage_data:
                    logger.info(f"  🔄 Arbitrage opportunities detected")
                    logger.info(f"    Best exchange: {arbitrage_data.get('best_exchange')}")
                    logger.info(f"    Price spread: ${arbitrage_data.get('price_spread', 0):.4f}")
                
                logger.info(f"  🤖 AI Response: {answer[:100]}...")
                
                self.results["passed"] += 1
                return True
            else:
                logger.error(f"❌ Single asset query failed: {response.get('error', 'Unknown error')}")
                self.results["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Single asset query test failed: {str(e)}")
            self.results["errors"].append(f"Single asset query: {str(e)}")
            self.results["failed"] += 1
            return False
    
    async def test_multi_asset_query(self):
        """Test multi-asset data collection"""
        logger.info("\n📊 Testing Multi-Asset Query...")
        
        try:
            query = "Compare the prices of BTC, ETH, and SOL"
            
            logger.info(f"Query: '{query}'")
            response = await self.agent.process_query(query)
            
            if response.get("status") == "success":
                data = response.get("data", {})
                answer = response.get("answer", "")
                
                successful_symbols = data.get("successful_symbols", [])
                assets_data = data.get("assets", {})
                
                logger.info(f"✅ Multi-asset query processed successfully")
                logger.info(f"  📈 Symbols processed: {len(successful_symbols)}")
                
                for symbol in successful_symbols:
                    asset_data = assets_data.get(symbol, {})
                    exchange_used = asset_data.get("exchange_used", "unknown")
                    current_price = asset_data.get("current_price", 0)
                    
                    logger.info(f"    {symbol}: ${current_price} (via {exchange_used})")
                
                logger.info(f"  🤖 AI Response: {answer[:100]}...")
                
                self.results["passed"] += 1
                return True
            else:
                logger.error(f"❌ Multi-asset query failed: {response.get('error', 'Unknown error')}")
                self.results["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Multi-asset query test failed: {str(e)}")
            self.results["errors"].append(f"Multi-asset query: {str(e)}")
            self.results["failed"] += 1
            return False
    
    async def test_best_price_comparison(self):
        """Test best price comparison across exchanges"""
        logger.info("\n🏆 Testing Best Price Comparison...")
        
        try:
            results = await self.agent.find_best_prices(self.test_symbols)
            
            if results.get("status") == "success":
                price_results = results.get("results", {})
                
                logger.info(f"✅ Best price comparison completed for {len(price_results)} symbols")
                
                for symbol, price_data in price_results.items():
                    if "error" not in price_data:
                        best_exchange = price_data.get("best_exchange", "unknown")
                        best_price = price_data.get("best_price", 0)
                        price_spread = price_data.get("price_spread", 0)
                        arbitrage_opportunity = price_data.get("arbitrage_opportunity", False)
                        
                        logger.info(f"  {symbol}:")
                        logger.info(f"    Best price: ${best_price} on {best_exchange}")
                        logger.info(f"    Price spread: ${price_spread:.4f}")
                        logger.info(f"    Arbitrage opportunity: {'Yes' if arbitrage_opportunity else 'No'}")
                    else:
                        logger.warning(f"  {symbol}: {price_data['error']}")
                
                self.results["passed"] += 1
                return True
            else:
                logger.error(f"❌ Best price comparison failed: {results.get('error', 'Unknown error')}")
                self.results["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Best price comparison test failed: {str(e)}")
            self.results["errors"].append(f"Best price comparison: {str(e)}")
            self.results["failed"] += 1
            return False
    
    async def test_exchange_failover(self):
        """Test exchange failover mechanism"""
        logger.info("\n🔄 Testing Exchange Failover...")
        
        try:
            # Test with a symbol that might not be available on all exchanges
            test_symbol = "BTCUSDT"
            
            # Get market data with specific exchange preferences
            market_data = await self.agent.exchange_aggregator.get_market_data(
                test_symbol,
                preferred_exchanges=["binance", "kucoin"],
                fallback_enabled=True
            )
            
            if market_data:
                logger.info(f"✅ Failover mechanism working")
                logger.info(f"  Symbol: {market_data.symbol}")
                logger.info(f"  Exchange used: {market_data.exchange}")
                logger.info(f"  Price: ${market_data.current_price}")
                
                self.results["passed"] += 1
                return True
            else:
                logger.error("❌ Failover mechanism failed - no data returned")
                self.results["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Exchange failover test failed: {str(e)}")
            self.results["errors"].append(f"Exchange failover: {str(e)}")
            self.results["failed"] += 1
            return False
    
    def print_summary(self):
        """Print test summary"""
        total_tests = self.results["passed"] + self.results["failed"]
        success_rate = (self.results["passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("🧪 MULTI-EXCHANGE INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        if self.results["errors"]:
            print(f"\n🚨 Errors encountered:")
            for error in self.results["errors"]:
                print(f"  • {error}")
        
        print("="*60)
        
        if success_rate >= 80:
            print("🎉 Multi-exchange integration is working well!")
        elif success_rate >= 60:
            print("⚠️ Multi-exchange integration has some issues but is functional")
        else:
            print("🚨 Multi-exchange integration needs attention")
        
        return success_rate >= 80

async def main():
    """Run the multi-exchange integration tests"""
    test_suite = MultiExchangeIntegrationTest()
    
    # Setup
    if not await test_suite.setup():
        print("❌ Test setup failed. Exiting.")
        return False
    
    # Run tests
    tests = [
        test_suite.test_exchange_registration,
        test_suite.test_exchange_health,
        test_suite.test_single_asset_query,
        test_suite.test_multi_asset_query,
        test_suite.test_best_price_comparison,
        test_suite.test_exchange_failover
    ]
    
    for test in tests:
        try:
            await test()
            await asyncio.sleep(1)  # Brief pause between tests
        except Exception as e:
            logger.error(f"Test execution error: {str(e)}")
            test_suite.results["failed"] += 1
            test_suite.results["errors"].append(f"Test execution: {str(e)}")
    
    # Print summary
    return test_suite.print_summary()

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1) 