"""
raf_service.py

Resource Allocation Function (RAF).

The RAF acts as the discovery/resource-allocation service for Sensing Units. It
is TAC-oriented rather than area-oriented: a radio TAC may contain multiple SUs.
The RAF contacts all available SUs in the requested TAC and aggregates CSI frames.

Default port: 8200
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PORT = int(os.getenv("PORT", "8200"))

# Multiple SUs can be mapped to the same TAC. Start only the instances needed for
# your local scenario. Unreachable SUs are reported as skipped units.
SUS_CONFIG = [
    {"suId": "SU-1", "radioTac": "226010001", "baseUrl": os.getenv("SU1_BASE_URL", "http://localhost:8101")},
    {"suId": "SU-2", "radioTac": "226010001", "baseUrl": os.getenv("SU2_BASE_URL", "http://localhost:8102")},
    {"suId": "SU-3", "radioTac": "226010002", "baseUrl": os.getenv("SU3_BASE_URL", "http://localhost:8103")},
]


class CSISample(BaseModel):
    bin: int
    ls_re: float
    ls_im: float


class CSIFrame(BaseModel):
    timestamp: str
    suId: str
    radioTac: str
    samples: List[CSISample]


class SensingRequestRAF(BaseModel):
    radioTac: str
    numSamples: int = Field(1, ge=1, le=100)


class SkippedSU(BaseModel):
    suId: str
    radioTac: str
    reason: str
    statusCode: Optional[int] = None


class CSIResponse(BaseModel):
    radioTac: str
    frames: List[CSIFrame]
    contributingSUs: List[str]
    skippedSUs: List[SkippedSU]


class Capability(BaseModel):
    suId: str
    radioTac: str
    isUavBased: bool
    sensingAvailable: bool
    currentWaveform: str
    supportedFunctions: List[str]
    sensingLogFile: str | None = None


app = FastAPI(
    title="ISAC Resource Allocation Function",
    version="0.3.0",
    description="TAC-based RAF that discovers SUs and aggregates CSI measurements.",
)


def _sus_for_tac(radio_tac: str) -> List[dict]:
    return [su for su in SUS_CONFIG if su["radioTac"] == radio_tac]


@app.get("/capabilities", response_model=Dict[str, Capability])
def list_capabilities():
    capabilities: Dict[str, Capability] = {}
    for su in SUS_CONFIG:
        try:
            resp = requests.get(su["baseUrl"].rstrip("/") + "/capabilities", timeout=3)
            resp.raise_for_status()
            data = resp.json()
            capabilities[su["suId"]] = Capability(
                suId=data["suId"],
                radioTac=data["radioTac"],
                isUavBased=data["isUavBased"],
                sensingAvailable=data["sensingAvailable"],
                currentWaveform=data["currentWaveform"],
                supportedFunctions=data.get("supportedFunctions", []),
                sensingLogFile=data.get("sensingLogFile"),
            )
        except Exception:
            # Discovery is best-effort; unreachable SUs are omitted here.
            continue
    return capabilities


@app.post("/measurements", response_model=CSIResponse)
def get_measurements(req: SensingRequestRAF):
    sus = _sus_for_tac(req.radioTac)
    if not sus:
        raise HTTPException(status_code=404, detail=f"No SUs registered for radio TAC {req.radioTac}")

    frames: List[CSIFrame] = []
    contributing: List[str] = []
    skipped: List[SkippedSU] = []

    for su in sus:
        try:
            resp = requests.post(
                su["baseUrl"].rstrip("/") + "/csi",
                json={"numFrames": req.numSamples},
                timeout=10,
            )
            if resp.status_code == 423:
                skipped.append(
                    SkippedSU(
                        suId=su["suId"],
                        radioTac=su["radioTac"],
                        reason="SU suspended sensing, most likely due to power-transfer mode.",
                        statusCode=423,
                    )
                )
                continue
            resp.raise_for_status()
            data = resp.json()
            for frame in data.get("frames", []):
                frames.append(CSIFrame(**frame))
            contributing.append(su["suId"])
        except Exception as exc:
            skipped.append(
                SkippedSU(
                    suId=su["suId"],
                    radioTac=su["radioTac"],
                    reason=str(exc),
                )
            )

    if not frames:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "No CSI frames available for the requested TAC.",
                "skippedSUs": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in skipped],
            },
        )

    return CSIResponse(
        radioTac=req.radioTac,
        frames=frames,
        contributingSUs=contributing,
        skippedSUs=skipped,
    )


@app.get("/healthz")
def healthcheck():
    return {
        "status": "ok",
        "registeredSUs": [su["suId"] for su in SUS_CONFIG],
        "radioTacs": sorted({su["radioTac"] for su in SUS_CONFIG}),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("raf_service:app", host="0.0.0.0", port=PORT, reload=False)
