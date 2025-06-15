#!/usr/bin/env python3
"""
Simple Agent Test
Debug the agent integration issue
"""

import asyncio
import sys
import os
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.core.ai.agent import MarketAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("AgentTest")

async def test_agent_simple():
    """Test the agent with a simple query"""
    
    print("🧪 Testing MarketAgent...")
    
    try:
        agent = MarketAgent()
        print("✅ Agent created successfully")
        
        # Test exchange health
        print("\n🔍 Testing exchange health...")
        health = await agent.get_exchange_health()
        print(f"Health data: {health}")
        
        # Test simple query
        print("\n🔍 Testing simple query...")
        query = "What is the price of Bitcoin?"
        
        try:
            response = await agent.process_query(query)
            print(f"✅ Query response: {response}")
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Agent creation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_simple()) 