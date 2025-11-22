# raf_service.py
# Resource Allocation Function – discovery + CSI aggregation from SUs.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import requests

# ---------- Config: where are the SUs? ----------

SUS_CONFIG = [
    {
        "suId": "SU-1",
        "areaId": "room-101",
        "baseUrl": "http://localhost:8101",
    },
    # You can add more SUs here later.
]


# ---------- Data models (mirroring SU) ----------

class CSISample(BaseModel):
    bin: int
    ls_re: float
    ls_im: float


class CSIFrame(BaseModel):
    timestamp: str
    suId: str
    samples: List[CSISample]


class CSIResponse(BaseModel):
    frames: List[CSIFrame]


class SensingRequestRAF(BaseModel):
    areaId: str
    suMode: int = Field(..., ge=1, le=3)
    numSamples: int = Field(1, ge=1, le=50)


class Capability(BaseModel):
    suId: str
    areaId: str
    modes: List[int]
    numBins: int


# ---------- RAF implementation ----------

app = FastAPI(
    title="ISAC Resource Allocation Function",
    version="0.1.0",
    description="RAF that discovers SUs and aggregates CSI measurements."
)


def _sus_for_area(area_id: str):
    return [su for su in SUS_CONFIG if su["areaId"] == area_id]


@app.get("/capabilities", response_model=Dict[str, Capability])
def list_capabilities():
    """Query all SUs for their capabilities."""
    caps: Dict[str, Capability] = {}
    for su in SUS_CONFIG:
        url = su["baseUrl"].rstrip("/") + "/capabilities"
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        caps[su["suId"]] = Capability(**data)
    return caps


@app.post("/measurements", response_model=CSIResponse)
def get_measurements(req: SensingRequestRAF):
    sus = _sus_for_area(req.areaId)
    if not sus:
        raise HTTPException(status_code=404, detail=f"No SUs for area {req.areaId}")

    all_frames: List[CSIFrame] = []

    for su in sus:
        url = su["baseUrl"].rstrip("/") + "/csi"
        payload = {"mode": req.suMode, "numFrames": req.numSamples}
        r = requests.post(url, json=payload, timeout=5)
        r.raise_for_status()
        data = r.json()
        for f in data.get("frames", []):
            all_frames.append(CSIFrame(**f))

    return CSIResponse(frames=all_frames)


@app.get("/healthz")
def healthcheck():
    return {"status": "ok", "registeredSUs": [su["suId"] for su in SUS_CONFIG]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("raf_service:app", host="0.0.0.0", port=8200, reload=False)
