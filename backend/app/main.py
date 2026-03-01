"""
main.py
-------
FastAPI application for flight delay prediction.

Environment variables
---------------------
ARTIFACTS_DIR  Path to model artifacts (default: /app/artifacts)
LOG_LEVEL      Logging level string     (default: INFO)
CORS_ORIGINS   Comma-separated origins  (default: *)
HOST           Uvicorn bind host        (default: 0.0.0.0)
PORT           Uvicorn bind port        (default: 8000)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import model as model_module
from . import features
from .schemas import ErrorResponse, FactorItem, HealthResponse, PredictRequest, PredictResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_raw_origins = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS: list[str] = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)


# ---------------------------------------------------------------------------
# Lifespan — warm up model at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading flight delay model...")
    try:
        mdl = model_module.get_model()
        if mdl.is_loaded:
            logger.info("Model ready. Version: %s", mdl.get_version())
        else:
            logger.warning("Model failed to load. /predict will return 503.")
    except Exception as exc:
        logger.error("Model warm-up exception: %s", exc, exc_info=True)
    yield
    logger.info("Shutting down Flight Delay Prediction API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    description=(
        "Predicts the probability that a commercial flight will be delayed by "
        "≥15 minutes, using a trained LightGBM classifier with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1_000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, latency_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(
        f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in exc.errors()
    )
    logger.warning("Validation error on %s: %s", request.url.path, detail)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="Validation Error", detail=detail).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="Bad Request", detail=str(exc)).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=f"HTTP {exc.status_code}", detail=str(exc.detail)).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Return service health and model readiness."""
    mdl = model_module.get_model()
    return HealthResponse(
        status="ok" if mdl.is_loaded else "degraded",
        model_loaded=mdl.is_loaded,
        version=mdl.get_version(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(request_body: PredictRequest) -> PredictResponse:
    """Predict flight delay probability with SHAP feature explanations."""
    t_start = time.perf_counter()
    request_hash = hashlib.sha256(request_body.model_dump_json().encode()).hexdigest()[:16]

    mdl = model_module.get_model()
    if not mdl.is_loaded:
        logger.error("[%s] Predict called but model is not loaded.", request_hash)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="Service Unavailable",
                detail="The prediction model is not loaded.",
            ).model_dump(),
        )

    raw_df = features.request_to_raw_df(request_body)
    probability, prediction, risk_level, shap_factors = mdl.predict(raw_df)

    latency_ms = (time.perf_counter() - t_start) * 1_000
    logger.info(
        "[%s] prediction=%d prob=%.4f risk=%s latency=%.1fms",
        request_hash, prediction, probability, risk_level, latency_ms,
    )
    if latency_ms > 200:
        logger.warning("[%s] Latency %.1fms exceeded 200ms target.", request_hash, latency_ms)

    return PredictResponse(
        delay_probability=round(probability, 4),
        prediction=prediction,
        risk_level=risk_level,
        top_factors=[FactorItem(**f) for f in shap_factors],
        threshold_used=mdl.threshold,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        reload=False,
    )
