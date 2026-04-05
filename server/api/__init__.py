from fastapi import APIRouter

from accordis.server.api.v1 import api_router_v1

api_router = APIRouter()

# v1 Router
api_router.include_router(api_router_v1)

__all__ = ["api_router"]