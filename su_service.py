# su_service.py
# Sensing Unit (radio-side) – generates CSI frames from the CSV datasets.

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import random

import pandas as pd

# ---------- Data models ----------

class CSISample(BaseModel):
    bin: int
    ls_re: float
    ls_im: float


class CSIFrame(BaseModel):
    timestamp: str
    suId: str
    samples: List[CSISample]


class CSIRequest(BaseModel):
    mode: int = Field(..., ge=1, le=3, description="1=human, 2=no-human, 3=variable")
    numFrames: int = Field(1, ge=1, le=50)


class CSIResponse(BaseModel):
    frames: List[CSIFrame]


class Capability(BaseModel):
    suId: str
    areaId: str
    modes: List[int]
    numBins: int


# ---------- Sensing Unit implementation ----------

class SensingUnit:
    """
    SU backed by two CSV files:
      1-short.csv -> human presence
      2-short.csv -> no human presence
    """

    def __init__(self, su_id: str, area_id: str,
                 df_human: pd.DataFrame, df_nohuman: pd.DataFrame) -> None:
        self.su_id = su_id
        self.area_id = area_id

        self._frames_human = list(df_human.groupby("timestamp"))
        self._frames_nohuman = list(df_nohuman.groupby("timestamp"))

        self.capability = Capability(
            suId=self.su_id,
            areaId=self.area_id,
            modes=[1, 2, 3],
            numBins=int(df_human["bin"].nunique())
        )

    def _pick_frame(self, mode: int):
        if mode == 1:
            ts, frame = random.choice(self._frames_human)
        elif mode == 2:
            ts, frame = random.choice(self._frames_nohuman)
        elif mode == 3:
            # Variable: randomly pick one of the two
            if random.random() < 0.5:
                ts, frame = random.choice(self._frames_human)
            else:
                ts, frame = random.choice(self._frames_nohuman)
        else:
            raise ValueError(f"Unsupported mode {mode}")
        return ts, frame

    def generate_frame(self, mode: int) -> CSIFrame:
        ts, frame = self._pick_frame(mode)
        samples = [
            CSISample(
                bin=int(row["bin"]),
                ls_re=float(row["ls_re"]),
                ls_im=float(row["ls_im"])
            )
            for _, row in frame.iterrows()
        ]
        return CSIFrame(
            timestamp=str(ts),
            suId=self.su_id,
            samples=samples
        )


# ---------- FastAPI wiring ----------

app = FastAPI(
    title="ISAC Sensing Unit",
    version="0.1.0",
    description="Sensing Unit generating CSI samples from CSV datasets."
)

# Load data and create a single SU instance (room-101, SU-1).
df_human = pd.read_csv("1-short.csv")
df_nohuman = pd.read_csv("2-short.csv")
SU_INSTANCE = SensingUnit("SU-1", "room-101", df_human, df_nohuman)


@app.get("/capabilities", response_model=Capability)
def get_capabilities():
    return SU_INSTANCE.capability


@app.post("/csi", response_model=CSIResponse)
def get_csi(req: CSIRequest):
    frames = [SU_INSTANCE.generate_frame(req.mode) for _ in range(req.numFrames)]
    return CSIResponse(frames=frames)


@app.get("/healthz")
def healthcheck():
    return {"status": "ok", "suId": SU_INSTANCE.su_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("su_service:app", host="0.0.0.0", port=8101, reload=False)
