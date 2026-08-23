"""
app.py — Phase 5: Predictive Maintenance API

Serves the trained model_bundle.pkl via FastAPI.
Run with: uvicorn app:app --reload --port 8000
Docs at:  http://127.0.0.1:8000/docs
"""

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from inference import load_bundle, predict

# ---------------------------------------------------------------------------
# Logging (prediction log — useful later for drift monitoring)
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)
logger = logging.getLogger("predictive_maintenance")

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Scores live sensor readings for failure risk. This is an early-warning "
        "classifier — it flags readings that look like failure states based on "
        "current sensor values, not a remaining-useful-life forecast."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    type: Literal["H", "L", "M"] = Field(..., alias="Type", description="Product quality type")
    air_temperature_k: float = Field(..., alias="Air temperature [K]", gt=0)
    process_temperature_k: float = Field(..., alias="Process temperature [K]", gt=0)
    rotational_speed_rpm: float = Field(..., alias="Rotational speed [rpm]", gt=0)
    torque_nm: float = Field(..., alias="Torque [Nm]", ge=0)
    tool_wear_min: float = Field(..., alias="Tool wear [min]", ge=0)

    model_config = {"populate_by_name": True}

    @field_validator("process_temperature_k")
    @classmethod
    def process_temp_reasonable(cls, v, info):
        # process temp is generally >= air temp for this dataset; sanity check only
        return v


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    risk_level: str
    threshold_used: float
    model_name: str
    timestamp: str


# ---------------------------------------------------------------------------
# Startup: load the bundle once so first request isn't slow
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _warm_up():
    load_bundle()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    bundle = load_bundle()
    return {
        "status": "ok",
        "model_name": bundle.get("model_name", "unknown"),
        "threshold": round(float(bundle["threshold"]), 4),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_failure(reading: SensorReading):
    raw = reading.model_dump(by_alias=True)
    try:
        result = predict(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "input=%s | probability=%.4f | prediction=%d | risk=%s",
        raw, result["failure_probability"], result["prediction"], result["risk_level"],
    )

    return result
