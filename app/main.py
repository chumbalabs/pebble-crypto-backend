from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import os
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes import health, market_data, predictions, websockets, ai_agent

# Load environment variables
load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CryptoPredictAPI")

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="pebble-crypto-api",
    description="Advanced Crypto Analytics & Predictions",
    version="0.3.1",
    docs_url="/docs",
    redoc_url=None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class LoggingMiddleware:
    async def __call__(self, request, call_next):
        response = await call_next(request)
        logger.info(f"{request.method} {request.url} - Status: {response.status_code}")
        return response
app.middleware("http")(LoggingMiddleware())

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(market_data.router, tags=["Market Data"])
app.include_router(predictions.router, tags=["Predictions"])
app.include_router(websockets.router)
app.include_router(ai_agent.router, prefix="/api", tags=["AI Agent"])

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    ) 