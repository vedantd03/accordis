from fastapi import APIRouter

from accordis.server.api.v1 import api_router_v1

def setup_router() -> APIRouter:
    api_router = APIRouter()
    
    # v1 APIRouter
    api_router.include_router(api_router_v1)
    
    return api_router