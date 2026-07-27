"""
power_transfer_orchestration_service.py

Power Transfer Orchestration Function (PTOF) for UAV-based Sensing Units.

The function evaluates the GPS coordinates of UAV SUs against hard-coded passive
sensor locations. When the UAV enters the activation radius of a passive sensor,
the PTOF instructs the SU to suspend human-presence sensing and switch to a
waveform dedicated to simultaneous power transfer and communication.

Default port: 8450
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

PORT = int(os.getenv("PORT", "8450"))
POWER_WAVEFORM = os.getenv("POWER_WAVEFORM", "PTC-WAVEFORM-v1")
ISAC_WAVEFORM = os.getenv("ISAC_WAVEFORM", "ISAC-HUMAN-PRESENCE-v1")

# Hard-coded passive sensors. Coordinates are illustrative and can be replaced
# with real deployment coordinates.
PASSIVE_SENSORS = [
    {
        "sensorId": "PS-ORO-001",
        "latitude": 44.435280,
        "longitude": 26.102950,
        "activationRadiusMeters": 30.0,
    },
    {
        "sensorId": "PS-ORO-002",
        "latitude": 44.436000,
        "longitude": 26.104100,
        "activationRadiusMeters": 25.0,
    },
]


class GpsPosition(BaseModel):
    latitude: float
    longitude: float
    altitudeMeters: float = 0.0


class PowerTransferEvaluateRequest(BaseModel):
    suId: str
    radioTac: str
    gps: GpsPosition


class PassiveSensorDistance(BaseModel):
    sensorId: str
    distanceMeters: float
    activationRadiusMeters: float


class PowerTransferDecision(BaseModel):
    suId: str
    radioTac: str
    timestamp: str
    powerTransferActive: bool
    sensingSuspended: bool
    selectedWaveform: str
    nearestSensor: Optional[PassiveSensorDistance]
    reason: str


class PassiveSensor(BaseModel):
    sensorId: str
    latitude: float
    longitude: float
    activationRadiusMeters: float


_LAST_DECISIONS: Dict[str, PowerTransferDecision] = {}


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_m * c


def _nearest_sensor(gps: GpsPosition) -> Optional[PassiveSensorDistance]:
    nearest = None
    for sensor in PASSIVE_SENSORS:
        distance = _haversine_meters(
            gps.latitude,
            gps.longitude,
            sensor["latitude"],
            sensor["longitude"],
        )
        candidate = PassiveSensorDistance(
            sensorId=sensor["sensorId"],
            distanceMeters=round(distance, 2),
            activationRadiusMeters=float(sensor["activationRadiusMeters"]),
        )
        if nearest is None or candidate.distanceMeters < nearest.distanceMeters:
            nearest = candidate
    return nearest


app = FastAPI(
    title="ISAC Power Transfer Orchestration Function",
    version="0.2.0",
    description="Coordinates UAV-based SU waveform switching for passive-sensor power transfer.",
)


@app.post("/power-transfer/evaluate", response_model=PowerTransferDecision)
def evaluate_power_transfer(req: PowerTransferEvaluateRequest):
    nearest = _nearest_sensor(req.gps)
    active = nearest is not None and nearest.distanceMeters <= nearest.activationRadiusMeters

    if active:
        waveform = POWER_WAVEFORM
        reason = f"UAV SU is within {nearest.activationRadiusMeters} m of passive sensor {nearest.sensorId}."
    else:
        waveform = ISAC_WAVEFORM
        reason = "UAV SU is outside the activation radius of all passive sensors."

    decision = PowerTransferDecision(
        suId=req.suId,
        radioTac=req.radioTac,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        powerTransferActive=active,
        sensingSuspended=active,
        selectedWaveform=waveform,
        nearestSensor=nearest,
        reason=reason,
    )
    _LAST_DECISIONS[req.suId] = decision
    return decision


@app.get("/passive-sensors", response_model=List[PassiveSensor])
def list_passive_sensors():
    return [PassiveSensor(**sensor) for sensor in PASSIVE_SENSORS]


@app.get("/uav-sus/{su_id}/decision", response_model=PowerTransferDecision)
def get_last_decision(su_id: str):
    return _LAST_DECISIONS[su_id]


@app.get("/healthz")
def healthcheck():
    return {
        "status": "ok",
        "passiveSensors": [sensor["sensorId"] for sensor in PASSIVE_SENSORS],
        "powerWaveform": POWER_WAVEFORM,
        "isacWaveform": ISAC_WAVEFORM,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("power_transfer_orchestration_service:app", host="0.0.0.0", port=PORT, reload=False)
