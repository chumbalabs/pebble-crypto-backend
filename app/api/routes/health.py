from fastapi import APIRouter, Request
from datetime import datetime, timezone
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/health", tags=["Health"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    return {
        "name": "pebble-crypto-api",
        "status": "online",
        "version": "0.3.1",
        "timestamp": datetime.now(timezone.utc).isoformat()
    } 