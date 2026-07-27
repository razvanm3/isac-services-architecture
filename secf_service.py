"""
secf_service.py

Sensing Control Function (SeCF).

The SeCF receives sanitized sensing requests from the Exposure Function using a
radio TAC, requests CSI frames from the RAF, forwards them to the SPF, and
applies quality-aware topology control based on the ML uncertainty.

Default port: 8400
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RAF_BASE_URL = os.getenv("RAF_BASE_URL", "http://localhost:8200")
SPF_BASE_URL = os.getenv("SPF_BASE_URL", "http://localhost:8300")
PORT = int(os.getenv("PORT", "8400"))
TOPOLOGY_UNCERTAINTY_THRESHOLD = float(os.getenv("TOPOLOGY_UNCERTAINTY_THRESHOLD", "40.0"))

CURRENT_TOPOLOGY = "monostatic"


class SensingControlRequest(BaseModel):
    radioTac: str
    numSamples: int = Field(1, ge=1, le=100)


class HumanPresenceResult(BaseModel):
    timestamp: str
    sourceTimestamp: str
    sourceSensingUnit: str
    radioTac: str
    humanPresence: bool
    uncertaintyPercent: float
    modelId: str


class SkippedSU(BaseModel):
    suId: str
    radioTac: str
    reason: str
    statusCode: Optional[int] = None


class SensingControlResponse(BaseModel):
    radioTac: str
    topologySwitched: bool
    currentTopology: str
    averageUncertaintyPercent: float
    aggregateHumanPresence: bool
    contributingSUs: List[str]
    skippedSUs: List[SkippedSU]
    results: List[HumanPresenceResult]


app = FastAPI(
    title="ISAC Sensing Control Function",
    version="0.2.0",
    description="SeCF coordinating RAF and SPF using radio TAC based requests.",
)


@app.post("/sensing-requests", response_model=SensingControlResponse)
def handle_sensing_request(req: SensingControlRequest):
    global CURRENT_TOPOLOGY

    try:
        raf_resp = requests.post(
            RAF_BASE_URL.rstrip("/") + "/measurements",
            json={"radioTac": req.radioTac, "numSamples": req.numSamples},
            timeout=15,
        )
        if raf_resp.status_code == 409:
            raise HTTPException(status_code=409, detail=raf_resp.json().get("detail", raf_resp.text))
        raf_resp.raise_for_status()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAF error: {exc}")

    raf_data = raf_resp.json()
    frames = raf_data.get("frames", [])
    if not frames:
        raise HTTPException(status_code=409, detail="RAF returned no CSI frames")

    try:
        spf_resp = requests.post(
            SPF_BASE_URL.rstrip("/") + "/process-csi",
            json={"frames": frames},
            timeout=20,
        )
        spf_resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SPF error: {exc}")

    results = [HumanPresenceResult(**item) for item in spf_resp.json().get("results", [])]
    if not results:
        raise HTTPException(status_code=502, detail="SPF returned no results")

    average_uncertainty = sum(r.uncertaintyPercent for r in results) / len(results)
    aggregate_human_presence = any(r.humanPresence for r in results)

    topology_switched = False
    if average_uncertainty > TOPOLOGY_UNCERTAINTY_THRESHOLD:
        if CURRENT_TOPOLOGY != "multistatic":
            CURRENT_TOPOLOGY = "multistatic"
            topology_switched = True
    else:
        if CURRENT_TOPOLOGY != "monostatic":
            CURRENT_TOPOLOGY = "monostatic"
            topology_switched = True

    return SensingControlResponse(
        radioTac=req.radioTac,
        topologySwitched=topology_switched,
        currentTopology=CURRENT_TOPOLOGY,
        averageUncertaintyPercent=round(average_uncertainty, 3),
        aggregateHumanPresence=aggregate_human_presence,
        contributingSUs=raf_data.get("contributingSUs", []),
        skippedSUs=[SkippedSU(**item) for item in raf_data.get("skippedSUs", [])],
        results=results,
    )


@app.get("/healthz")
def healthcheck():
    return {
        "status": "ok",
        "currentTopology": CURRENT_TOPOLOGY,
        "topologyUncertaintyThreshold": TOPOLOGY_UNCERTAINTY_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("secf_service:app", host="0.0.0.0", port=PORT, reload=False)
