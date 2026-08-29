"""
AI-NIDS — API Schemas
PHASE 11
Pydantic request/response models for FastAPI.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request containing the 50 selected network-flow features.

    The API accepts features as:
        {
            "features": {
                "Destination Port": 80,
                ...
            }
        }
    """

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing the 50 selected network-flow features."
    )


class PredictionResponse(BaseModel):
    """
    Two-stage AI-NIDS prediction response.
    """

    is_attack: bool = Field(
        ...,
        description="True if the binary model detects an attack."
    )

    prediction: str = Field(
        ...,
        description="BENIGN or ATTACK."
    )

    attack_type: str = Field(
        ...,
        description="Predicted attack type. BENIGN when no attack is detected."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final prediction confidence."
    )

    binary_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Binary model confidence."
    )

    attack_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Multi-class attack confidence. Null for BENIGN traffic."
    )


class HealthResponse(BaseModel):
    """
    API health response.
    """

    status: str
    service: str
    binary_model_loaded: bool
    multiclass_model_loaded: bool
    feature_count: int


class ModelInfoResponse(BaseModel):
    """
    Information about the loaded AI-NIDS models.
    """

    binary_model: str
    multiclass_model: str
    feature_count: int
    attack_classes: list[str]
    architecture: str