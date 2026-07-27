"""
exposure_service.py

Exposure Function (EF) exposing a CAMARA-style client-facing API for ISAC human
presence sensing.

The client first obtains a bearer token from /oauth2/token. The authentication
profile embedded in the token contains the client-id, allowed radio TACs and
scopes. The sensing endpoint derives the client identity from the token and does
not accept clientId in the request body.

Default port: 8500
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

SECF_BASE_URL = os.getenv("SECF_BASE_URL", "http://localhost:8400")
PORT = int(os.getenv("PORT", "8500"))
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "dev-only-change-this-secret")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))

# Demo authentication profiles. In a production implementation this would be
# externalized to an IAM/IdP and enforced through OAuth2/OIDC scopes/claims.
CLIENT_AUTH_PROFILES: Dict[str, dict] = {
    "client-A": {
        "clientSecret": "client-A-secret",
        "allowedRadioTacs": ["226010001"],
        "scopes": ["isac-human-presence:read"],
    },
    "client-B": {
        "clientSecret": "client-B-secret",
        "allowedRadioTacs": ["226010001", "226010002"],
        "scopes": ["isac-human-presence:read"],
    },
}


# -----------------------------------------------------------------------------
# API models
# -----------------------------------------------------------------------------


class TokenRequest(BaseModel):
    grantType: str = Field("client_credentials", pattern="^client_credentials$")
    clientId: str
    clientSecret: str
    scope: str = "isac-human-presence:read"


class TokenResponse(BaseModel):
    accessToken: str
    tokenType: str = "Bearer"
    expiresIn: int
    scope: str


class HumanPresenceRetrieveRequest(BaseModel):
    radioTac: str = Field(..., description="Radio Tracking Area Code serving the sensing request.")
    numSamples: int = Field(1, ge=1, le=100)
    maxAgeSeconds: Optional[int] = Field(None, ge=0, le=3600)


class HumanPresenceResult(BaseModel):
    timestamp: str
    sourceTimestamp: str
    sourceSensingUnit: str
    radioTac: str
    humanPresence: bool
    uncertaintyPercent: float
    modelId: str


class HumanPresenceRetrieveResponse(BaseModel):
    requestId: str
    clientId: str
    radioTac: str
    sensingResultTime: str
    humanPresence: bool
    confidencePercent: float
    averageUncertaintyPercent: float
    topology: dict
    sensingUnits: dict
    results: List[HumanPresenceResult]


# -----------------------------------------------------------------------------
# Token helpers
# -----------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str) -> str:
    signature = hmac.new(TOKEN_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(signature)


def _issue_token(client_id: str, scope: str) -> str:
    profile = CLIENT_AUTH_PROFILES[client_id]
    payload = {
        "clientId": client_id,
        "allowedRadioTacs": profile["allowedRadioTacs"],
        "scope": scope,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature_b64 = _sign(payload_b64)
    return f"{payload_b64}.{signature_b64}"


def _validate_token(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "", 1)
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        expected = _sign(payload_b64)
        if not hmac.compare_digest(signature_b64, expected):
            raise ValueError("signature mismatch")
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    if int(time.time()) > int(payload.get("exp", 0)):
        raise HTTPException(status_code=401, detail="Expired bearer token")
    return payload


def _authorize_tac_and_scope(token_payload: dict, radio_tac: str, required_scope: str) -> None:
    if required_scope not in token_payload.get("scope", "").split():
        raise HTTPException(status_code=403, detail=f"Missing required scope {required_scope}")
    if radio_tac not in token_payload.get("allowedRadioTacs", []):
        raise HTTPException(status_code=403, detail=f"Client is not authorized for radio TAC {radio_tac}")


# -----------------------------------------------------------------------------
# FastAPI wiring
# -----------------------------------------------------------------------------


app = FastAPI(
    title="ISAC Exposure Function",
    version="0.2.0",
    description="CAMARA-style API exposure for ISAC human-presence sensing.",
)


@app.post("/oauth2/token", response_model=TokenResponse, tags=["Authorization"])
def token(req: TokenRequest):
    profile = CLIENT_AUTH_PROFILES.get(req.clientId)
    if profile is None or profile["clientSecret"] != req.clientSecret:
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    if req.scope not in profile["scopes"]:
        raise HTTPException(status_code=403, detail="Requested scope is not allowed for client")
    access_token = _issue_token(req.clientId, req.scope)
    return TokenResponse(
        accessToken=access_token,
        expiresIn=TOKEN_TTL_SECONDS,
        scope=req.scope,
    )


@app.post(
    "/isac-human-presence/v0.1/retrieve",
    response_model=HumanPresenceRetrieveResponse,
    tags=["Human Presence Sensing"],
    summary="Retrieve human presence information for a radio TAC",
)
def retrieve_human_presence(req: HumanPresenceRetrieveRequest, authorization: Optional[str] = Header(default=None)):
    token_payload = _validate_token(authorization)
    _authorize_tac_and_scope(token_payload, req.radioTac, "isac-human-presence:read")

    try:
        secf_resp = requests.post(
            SECF_BASE_URL.rstrip("/") + "/sensing-requests",
            json={"radioTac": req.radioTac, "numSamples": req.numSamples},
            timeout=30,
        )
        if secf_resp.status_code == 409:
            raise HTTPException(status_code=409, detail=secf_resp.json().get("detail", secf_resp.text))
        secf_resp.raise_for_status()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SeCF error: {exc}")

    data = secf_resp.json()
    results = [HumanPresenceResult(**item) for item in data.get("results", [])]
    avg_unc = float(data.get("averageUncertaintyPercent", 100.0))
    confidence = max(0.0, min(100.0, 100.0 - avg_unc))

    return HumanPresenceRetrieveResponse(
        requestId=str(uuid.uuid4()),
        clientId=token_payload["clientId"],
        radioTac=req.radioTac,
        sensingResultTime=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        humanPresence=bool(data.get("aggregateHumanPresence", False)),
        confidencePercent=round(confidence, 3),
        averageUncertaintyPercent=avg_unc,
        topology={
            "currentTopology": data.get("currentTopology"),
            "topologySwitched": data.get("topologySwitched", False),
        },
        sensingUnits={
            "contributingSUs": data.get("contributingSUs", []),
            "skippedSUs": data.get("skippedSUs", []),
        },
        results=results,
    )


@app.get("/healthz")
def healthcheck():
    return {
        "status": "ok",
        "clientProfiles": list(CLIENT_AUTH_PROFILES.keys()),
        "secfBaseUrl": SECF_BASE_URL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("exposure_service:app", host="0.0.0.0", port=PORT, reload=False)
