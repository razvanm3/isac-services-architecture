"""
su_service.py

Sensing Unit (SU) service for the ISAC service-based architecture.

The SU is data-source agnostic. It does not call any external radio-log service
and it does not implement source-specific parsing logic. A raw radio log file is
provided through SENSING_LOG_FILE. The helper module sensing_unit_input_helper.py
transforms that log into normalized CSI input samples for the SU.

The SU exposes the normalized CSI frames to the RAF through POST /csi.

If IS_UAV_BASED=true, the SU attaches itself to the Power Transfer Orchestration
Function. A background loop periodically updates the UAV GPS position and asks
PTOF whether the SU should suspend human-presence sensing and switch to the
power-transfer-and-communication waveform.

Default port: 8101
"""

from __future__ import annotations

import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests

from sensing_unit_input_helper import parse_latest_csi_frames


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


SU_ID = os.getenv("SU_ID", "SU-1")
RADIO_TAC = os.getenv("RADIO_TAC", "226010001")
PORT = int(os.getenv("PORT", "8101"))

# The SU consumes an externally provided raw radio log file. The helper converts
# it into normalized CSI frames.
SENSING_LOG_FILE = os.getenv("SENSING_LOG_FILE", os.getenv("RADIO_LOG_FILE", "samples/srs_debug_small.log")).strip()
CSI_LS_SCALE = float(os.getenv("CSI_LS_SCALE", os.getenv("OAIBOX_LS_SCALE", "1.0")))

IS_UAV_BASED = _env_bool("IS_UAV_BASED", False)
PTOF_BASE_URL = os.getenv("PTOF_BASE_URL", "http://localhost:8450")
PTOF_CHECK_INTERVAL_SECONDS = float(os.getenv("PTOF_CHECK_INTERVAL_SECONDS", "2.0"))

INITIAL_LATITUDE = float(os.getenv("INITIAL_LATITUDE", "44.435200"))
INITIAL_LONGITUDE = float(os.getenv("INITIAL_LONGITUDE", "26.102800"))
INITIAL_ALTITUDE_METERS = float(os.getenv("INITIAL_ALTITUDE_METERS", "60.0"))

ISAC_WAVEFORM = os.getenv("ISAC_WAVEFORM", "ISAC-HUMAN-PRESENCE-v1")


# -----------------------------------------------------------------------------
# API models
# -----------------------------------------------------------------------------


class CSISample(BaseModel):
    bin: int
    ls_re: float
    ls_im: float


class CSIFrame(BaseModel):
    timestamp: str
    suId: str
    radioTac: str
    samples: List[CSISample]


class CSIRequest(BaseModel):
    numFrames: int = Field(1, ge=1, le=100)


class CSIResponse(BaseModel):
    frames: List[CSIFrame]


class GpsPosition(BaseModel):
    latitude: float
    longitude: float
    altitudeMeters: float


class SUCapability(BaseModel):
    suId: str
    radioTac: str
    isUavBased: bool
    sensingLogFile: str
    sensingAvailable: bool
    currentWaveform: str
    supportedFunctions: List[str]


class SUState(BaseModel):
    suId: str
    radioTac: str
    isUavBased: bool
    sensingLogFile: str
    sensingSuspended: bool
    currentWaveform: str
    gps: Optional[GpsPosition]
    lastPowerTransferDecision: Optional[dict]
    timestamp: str


class WaveformControlRequest(BaseModel):
    selectedWaveform: str
    sensingSuspended: bool
    reason: Optional[str] = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _pydantic_dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


# -----------------------------------------------------------------------------
# Sensing Unit runtime
# -----------------------------------------------------------------------------


class SensingUnitRuntime:
    def __init__(self) -> None:
        self.su_id = SU_ID
        self.radio_tac = RADIO_TAC
        self.sensing_log_file = SENSING_LOG_FILE
        self.current_waveform = ISAC_WAVEFORM
        self.is_uav_based = IS_UAV_BASED
        self.sensing_suspended = False
        self.last_power_transfer_decision: Optional[dict] = None
        self._lock = threading.Lock()
        self.gps = GpsPosition(
            latitude=INITIAL_LATITUDE,
            longitude=INITIAL_LONGITUDE,
            altitudeMeters=INITIAL_ALTITUDE_METERS,
        ) if self.is_uav_based else None

    def start_background_tasks(self) -> None:
        if not self.is_uav_based:
            return
        t = threading.Thread(target=self._uav_power_transfer_loop, daemon=True)
        t.start()

    def _simulate_uav_mobility_step(self) -> None:
        if self.gps is None:
            return
        self.gps.latitude += random.uniform(-0.000015, 0.000015)
        self.gps.longitude += random.uniform(-0.000020, 0.000020)

    def _uav_power_transfer_loop(self) -> None:
        while True:
            try:
                self._simulate_uav_mobility_step()
                if self.gps is not None:
                    payload = {
                        "suId": self.su_id,
                        "radioTac": self.radio_tac,
                        "gps": _pydantic_dump(self.gps),
                    }
                    resp = requests.post(
                        PTOF_BASE_URL.rstrip("/") + "/power-transfer/evaluate",
                        json=payload,
                        timeout=3,
                    )
                    resp.raise_for_status()
                    decision = resp.json()
                    with self._lock:
                        self.last_power_transfer_decision = decision
                        self.current_waveform = decision["selectedWaveform"]
                        self.sensing_suspended = bool(decision["sensingSuspended"])
            except Exception as exc:
                with self._lock:
                    self.last_power_transfer_decision = {"error": str(exc)}
            time.sleep(PTOF_CHECK_INTERVAL_SECONDS)

    def capability(self) -> SUCapability:
        with self._lock:
            return SUCapability(
                suId=self.su_id,
                radioTac=self.radio_tac,
                isUavBased=self.is_uav_based,
                sensingLogFile=self.sensing_log_file,
                sensingAvailable=not self.sensing_suspended,
                currentWaveform=self.current_waveform,
                supportedFunctions=[
                    "human-presence-sensing",
                    "csi-forwarding",
                    "raw-radio-log-to-csi-adapter",
                    *(["uav-power-transfer-orchestration"] if self.is_uav_based else []),
                ],
            )

    def state(self) -> SUState:
        with self._lock:
            return SUState(
                suId=self.su_id,
                radioTac=self.radio_tac,
                isUavBased=self.is_uav_based,
                sensingLogFile=self.sensing_log_file,
                sensingSuspended=self.sensing_suspended,
                currentWaveform=self.current_waveform,
                gps=self.gps,
                lastPowerTransferDecision=self.last_power_transfer_decision,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )

    def apply_waveform_control(self, req: WaveformControlRequest) -> SUState:
        with self._lock:
            self.current_waveform = req.selectedWaveform
            self.sensing_suspended = req.sensingSuspended
        return self.state()

    def _ensure_sensing_available(self) -> None:
        with self._lock:
            suspended = self.sensing_suspended
            waveform = self.current_waveform
            decision = self.last_power_transfer_decision

        if suspended:
            raise HTTPException(
                status_code=423,
                detail={
                    "message": "Human-presence sensing is suspended for this SU.",
                    "currentWaveform": waveform,
                    "powerTransferDecision": decision,
                },
            )

    def get_csi_frames(self, num_frames: int) -> List[CSIFrame]:
        self._ensure_sensing_available()

        frame_dicts = parse_latest_csi_frames(
            self.sensing_log_file,
            su_id=self.su_id,
            radio_tac=self.radio_tac,
            num_frames=num_frames,
            ls_scale=CSI_LS_SCALE,
        )

        if not frame_dicts:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "No complete CSI frames available in the sensing log file.",
                    "sensingLogFile": self.sensing_log_file,
                },
            )

        return [CSIFrame(**frame) for frame in frame_dicts]


RUNTIME = SensingUnitRuntime()

app = FastAPI(
    title="ISAC Sensing Unit",
    version="0.6.0",
    description="Data-source-agnostic Sensing Unit consuming helper-normalized CSI samples from a raw log file.",
)


@app.on_event("startup")
def _startup():
    RUNTIME.start_background_tasks()


@app.get("/capabilities", response_model=SUCapability)
def get_capabilities():
    return RUNTIME.capability()


@app.get("/state", response_model=SUState)
def get_state():
    return RUNTIME.state()


@app.post("/control/waveform", response_model=SUState)
def control_waveform(req: WaveformControlRequest):
    return RUNTIME.apply_waveform_control(req)


@app.post("/csi", response_model=CSIResponse)
def get_csi(req: CSIRequest):
    try:
        return CSIResponse(frames=RUNTIME.get_csi_frames(req.numFrames))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Sensing log input error: {exc}")


@app.get("/healthz")
def healthcheck():
    state = RUNTIME.state()
    return {
        "status": "ok",
        "suId": state.suId,
        "radioTac": state.radioTac,
        "sensingLogFile": state.sensingLogFile,
        "isUavBased": state.isUavBased,
        "sensingSuspended": state.sensingSuspended,
        "currentWaveform": state.currentWaveform,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("su_service:app", host="0.0.0.0", port=PORT, reload=False)
