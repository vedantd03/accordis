from fastapi import APIRouter

from accordis.server.api import api_router

def setup_router() -> APIRouter:
    router = APIRouter()
    
    # API Router
    router.include_router(api_router)
    
    return router