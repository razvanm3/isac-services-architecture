# spf_service.py
# Sensing Processing Function – ML model over CSI frames.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ---------- Data models (CSI) ----------

class CSISample(BaseModel):
    bin: int
    ls_re: float
    ls_im: float


class CSIFrame(BaseModel):
    timestamp: str
    suId: str
    samples: List[CSISample]


class SPFRequest(BaseModel):
    frames: List[CSIFrame]


class HumanPresenceResult(BaseModel):
    timestamp: str
    humanPresence: bool
    uncertaintyPercent: float


class SPFResponse(BaseModel):
    results: List[HumanPresenceResult]


# ---------- ML model ----------

class HumanPresenceModel:
    def __init__(self,
                 human_csv_path: str = "1-short.csv",
                 nohuman_csv_path: str = "2-short.csv") -> None:
        df_human = pd.read_csv(human_csv_path)
        df_nohuman = pd.read_csv(nohuman_csv_path)

        feat_human = self._build_features(df_human, label=1)
        feat_nohuman = self._build_features(df_nohuman, label=0)

        data = pd.concat([feat_human, feat_nohuman], ignore_index=True)

        self._feature_cols = [
            "ls_re_mean", "ls_re_std",
            "ls_im_mean", "ls_im_std",
            "mag_mean", "mag_std"
        ]

        X = data[self._feature_cols]
        y = data["label"]

        self._clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        self._clf.fit(X, y)

    @staticmethod
    def _build_features(df: pd.DataFrame, label: int) -> pd.DataFrame:
        feats = []
        for ts, group in df.groupby("timestamp"):
            mag = np.sqrt(group["ls_re"] ** 2 + group["ls_im"] ** 2)
            feats.append({
                "timestamp": ts,
                "ls_re_mean": group["ls_re"].mean(),
                "ls_re_std": group["ls_re"].std(),
                "ls_im_mean": group["ls_im"].mean(),
                "ls_im_std": group["ls_im"].std(),
                "mag_mean": mag.mean(),
                "mag_std": mag.std(),
                "label": label,
            })
        return pd.DataFrame(feats)

    def infer_from_csi(self, csi_frame: pd.DataFrame):
        mag = np.sqrt(csi_frame["ls_re"] ** 2 + csi_frame["ls_im"] ** 2)
        x = pd.DataFrame([{
            "ls_re_mean": csi_frame["ls_re"].mean(),
            "ls_re_std": csi_frame["ls_re"].std(),
            "ls_im_mean": csi_frame["ls_im"].mean(),
            "ls_im_std": csi_frame["ls_im"].std(),
            "mag_mean": mag.mean(),
            "mag_std": mag.std(),
        }])

        proba = self._clf.predict_proba(x)[0]
        presence_prob = float(proba[1])
        max_prob = float(max(proba))
        human_present = presence_prob >= 0.5
        uncertainty_percent = (1.0 - max_prob) * 100.0
        return human_present, uncertainty_percent


# ---------- FastAPI ----------

app = FastAPI(
    title="ISAC Sensing Processing Function",
    version="0.1.0",
    description="SPF running ML-based human presence detection on CSI frames."
)

MODEL = HumanPresenceModel()


@app.post("/process-csi", response_model=SPFResponse)
def process_csi(req: SPFRequest):
    results: List[HumanPresenceResult] = []

    for frame in req.frames:
        df = pd.DataFrame(
            [
                {
                    "timestamp": frame.timestamp,
                    "bin": s.bin,
                    "ls_re": s.ls_re,
                    "ls_im": s.ls_im,
                }
                for s in frame.samples
            ]
        )

        human_present, uncertainty = MODEL.infer_from_csi(df)
        ts_out = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        results.append(
            HumanPresenceResult(
                timestamp=ts_out,
                humanPresence=human_present,
                uncertaintyPercent=uncertainty,
            )
        )

    return SPFResponse(results=results)


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("spf_service:app", host="0.0.0.0", port=8300, reload=False)
