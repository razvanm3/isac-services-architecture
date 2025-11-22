# secf_service.py
# Sensing Control Function – orchestrates RAF + SPF and decides topology.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import requests

# URLs of internal services
RAF_BASE_URL = "http://localhost:8200"
SPF_BASE_URL = "http://localhost:8300"


class SensingControlRequest(BaseModel):
    areaId: str
    numSamples: int = Field(1, ge=1, le=50)
    suMode: int = Field(..., ge=1, le=3)


class HumanPresenceResult(BaseModel):
    timestamp: str
    humanPresence: bool
    uncertaintyPercent: float


class SensingControlResponse(BaseModel):
    topologySwitched: bool
    currentTopology: str
    results: List[HumanPresenceResult]


# simple in-memory topology state
CURRENT_TOPOLOGY = "monostatic"


app = FastAPI(
    title="ISAC Sensing Control Function",
    version="0.1.0",
    description="SeCF coordinating RAF and SPF, selecting ISAC topology."
)


@app.post("/sensing-requests", response_model=SensingControlResponse)
def handle_sensing_request(req: SensingControlRequest):
    global CURRENT_TOPOLOGY

    # 1) Ask RAF for CSI frames
    raf_payload = {
        "areaId": req.areaId,
        "suMode": req.suMode,
        "numSamples": req.numSamples,
    }
    try:
        r_raf = requests.post(
            RAF_BASE_URL.rstrip("/") + "/measurements",
            json=raf_payload,
            timeout=10,
        )
        r_raf.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAF error: {e}")

    csi_data = r_raf.json()

    # 2) Send CSI to SPF
    try:
        r_spf = requests.post(
            SPF_BASE_URL.rstrip("/") + "/process-csi",
            json={"frames": csi_data.get("frames", [])},
            timeout=20,
        )
        r_spf.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPF error: {e}")

    spf_results = r_spf.json().get("results", [])
    results = [HumanPresenceResult(**r) for r in spf_results]

    # 3) Decide if topology needs to switch based on avg uncertainty
    if not results:
        raise HTTPException(status_code=500, detail="No SPF results")

    avg_unc = sum(r.uncertaintyPercent for r in results) / len(results)

    topology_switched = False
    if avg_unc > 40.0:
        if CURRENT_TOPOLOGY != "multistatic":
            CURRENT_TOPOLOGY = "multistatic"
            topology_switched = True
    else:
        if CURRENT_TOPOLOGY != "monostatic":
            CURRENT_TOPOLOGY = "monostatic"
            topology_switched = True

    return SensingControlResponse(
        topologySwitched=topology_switched,
        currentTopology=CURRENT_TOPOLOGY,
        results=results,
    )


@app.get("/healthz")
def healthcheck():
    return {"status": "ok", "currentTopology": CURRENT_TOPOLOGY}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("secf_service:app", host="0.0.0.0", port=8400, reload=False)
