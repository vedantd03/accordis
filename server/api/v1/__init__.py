from fastapi import APIRouter

from accordis.server.api.v1.baseline import router as baseline_router

api_router_v1 = APIRouter()

api_router_v1.include_router(baseline_router)

__all__ = ["api_router_v1"]