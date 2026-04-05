from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from accordis.server.utils.baseline_helper import run_baseline

router = APIRouter(prefix="/baseline", tags=["baseline"])


class BaselineRequest(BaseModel):
    provider: Optional[str] = Field(
        default="openai",
        description="Inference provider. Options: 'static' (no LLM), 'openai', 'gemini'.",
        examples=["static", "openai", "gemini"],
    )
    tasks: Optional[list[str]] = Field(
        default=None,
        description=(
            "Task difficulties to evaluate. Defaults to all three if omitted: "
            "['easy', 'medium', 'hard']."
        ),
        examples=[["easy"], ["easy", "medium", "hard"]],
    )
    model: Optional[str] = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        description="LLM model name. Required when provider is 'openai' or 'gemini'.",
        examples=["gpt-5.4", "gemini-3.1-flash-lite-preview", "Qwen/Qwen2.5-72B-Instruct"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"provider": "static"},
                {"provider": "static", "tasks": ["easy", "medium"]},
                {"provider": "openai", "model": "gpt-4o", "tasks": ["easy"]},
            ]
        }
    }

    @model_validator(mode="after")
    def validate_provider_model(self):
        if self.provider in ("openai", "gemini") and self.model is None:
            raise ValueError(
                f"The 'model' field is required when provider is '{self.provider}'."
            )
        if self.tasks is not None:
            valid = {"easy", "medium", "hard"}
            invalid = set(self.tasks) - valid
            if invalid:
                raise ValueError(
                    f"Invalid task(s): {invalid}. Must be one of {valid}."
                )
        return self


@router.post("/")
async def baseline(baseline_request: Optional[BaselineRequest] = None):
    """Trigger baseline inference for one or more Accordis task difficulties."""
    if baseline_request is None:
        baseline_request = BaselineRequest()

    try:
        result = await run_baseline(
            provider=baseline_request.provider,
            model=baseline_request.model,
            tasks=baseline_request.tasks,
        )
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )
