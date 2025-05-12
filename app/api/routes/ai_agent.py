# AI agent endpoints

from fastapi import APIRouter, HTTPException, Request, Body
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Dict, Any
from app.core.ai.agent import MarketAgent

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
market_agent = MarketAgent()

@router.post("/ask", tags=["AI Agent"])
@limiter.limit("10/minute")
async def ask_agent(request: Request, query: Dict[str, str] = Body(...)):
    """
    Ask the AI agent a natural language question about cryptocurrency markets.
    
    Examples:
    - "What is the current price of BTC?"
    - "What is the trend for Ethereum?"
    - "Should I buy SOL now?"
    - "How volatile is LINK today?"
    
    Request body should contain a "question" field with the natural language query.
    """
    try:
        if "question" not in query or not query["question"].strip():
            raise HTTPException(status_code=400, detail="Question is required")
            
        # Process the query
        response = await market_agent.process_query(query["question"])
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
