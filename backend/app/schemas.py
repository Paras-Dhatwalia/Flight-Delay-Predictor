"""
schemas.py
----------
Pydantic v2 request/response models for the Flight Delay Prediction API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    airline: str = Field(
        ...,
        description="IATA airline code (2-3 chars), e.g. 'AA'",
        min_length=2,
        max_length=3,
    )
    origin: str = Field(
        ...,
        description="IATA origin airport code (3 chars), e.g. 'JFK'",
        min_length=3,
        max_length=4,
    )
    destination: str = Field(
        ...,
        description="IATA destination airport code (3 chars), e.g. 'LAX'",
        min_length=3,
        max_length=4,
    )
    scheduled_departure: datetime = Field(
        ...,
        description="Scheduled departure datetime, e.g. '2026-07-12T14:30:00'",
    )
    tail_number: Optional[str] = Field(
        None,
        description="Aircraft tail number (optional), e.g. 'N123AA'",
    )

    @field_validator("airline", "origin", "destination", mode="before")
    @classmethod
    def uppercase_strip(cls, v: str) -> str:
        return v.upper().strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "airline": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure": "2026-07-12T14:30:00",
                "tail_number": "N123AA",
            }
        }
    }


class FactorItem(BaseModel):
    feature: str
    impact: float


class PredictResponse(BaseModel):
    delay_probability: float = Field(..., description="P(delay >= 15 min) in [0, 1]")
    prediction: int           = Field(..., description="Binary prediction: 0=on-time, 1=delayed")
    risk_level: str           = Field(..., description="Low | Medium | High")
    top_factors: list[FactorItem] = Field(..., description="Top SHAP feature contributions")
    threshold_used: float     = Field(..., description="Decision threshold applied")


class HealthResponse(BaseModel):
    status: str       = Field(..., description="'ok' or 'degraded'")
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
