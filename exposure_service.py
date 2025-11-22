# exposure_service.py
# Exposure Function – CAMARA-style API exposed to external Sensing Clients.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import requests

SECF_BASE_URL = "http://localhost:8400"


# ---------- Authorization / policy ----------

class AuthorizationService:
    def __init__(self, allowed_areas_by_client: Dict[str, List[str]]) -> None:
        self._allowed = {cid: set(areas) for cid, areas in allowed_areas_by_client.items()}

    def check(self, client_id: str, area_id: str):
        areas = self._allowed.get(client_id)
        if areas is None or area_id not in areas:
            raise PermissionError(
                f"Client '{client_id}' is not allowed to request sensing in area '{area_id}'"
            )


AUTHZ = AuthorizationService(
    allowed_areas_by_client={
        "client-A": ["room-101"],
        "client-B": ["room-101", "room-102"],
    }
)


# ---------- External API models (CAMARA-like) ----------

class HumanPresenceSensingRequest(BaseModel):
    clientId: str
    areaId: str
    numSamples: int = Field(1, ge=1, le=50)
    suMode: int = Field(3, ge=1, le=3)


class HumanPresenceResult(BaseModel):
    timestamp: str
    humanPresence: bool
    uncertaintyPercent: float


class HumanPresenceSensingResponse(BaseModel):
    clientId: str
    areaId: str
    topologySwitched: bool
    currentTopology: str
    results: List[HumanPresenceResult]


# ---------- FastAPI ----------

app = FastAPI(
    title="ISAC Exposure Function",
    version="0.1.0",
    description="CAMARA-style human presence sensing API."
)


@app.post(
    "/isac/human-presence/v0.1/detect",
    response_model=HumanPresenceSensingResponse,
    summary="Human presence sensing (synchronous)"
)
def detect_human_presence(req: HumanPresenceSensingRequest):
    # 1) Authorization
    try:
        AUTHZ.check(req.clientId, req.areaId)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # 2) Forward to SeCF
    secf_payload = {
        "areaId": req.areaId,
        "numSamples": req.numSamples,
        "suMode": req.suMode,
    }
    try:
        r = requests.post(
            SECF_BASE_URL.rstrip("/") + "/sensing-requests",
            json=secf_payload,
            timeout=20,
        )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SeCF error: {e}")

    secf_data = r.json()

    results = [HumanPresenceResult(**r) for r in secf_data.get("results", [])]

    return HumanPresenceSensingResponse(
        clientId=req.clientId,
        areaId=req.areaId,
        topologySwitched=secf_data.get("topologySwitched", False),
        currentTopology=secf_data.get("currentTopology", "unknown"),
        results=results,
    )


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("exposure_service:app", host="0.0.0.0", port=8500, reload=False)
