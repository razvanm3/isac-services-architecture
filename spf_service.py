"""
spf_service.py

Sensing Processing Function (SPF).

The SPF receives CSI frames from the SeCF and applies the provided ML model for
human-presence detection. If MODEL_PATH points to a valid joblib file, that model
is loaded. Otherwise, the SPF trains a fallback Random Forest model from the
reference datasets.

Default port: 8300
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

PORT = int(os.getenv("PORT", "8300"))

# Prefer the packaged model name, but keep backwards compatibility with the
# previous README value models/human_presence_model.joblib.
MODEL_PATH = os.getenv("MODEL_PATH", "models/isac_human_presence.joblib")
HUMAN_DATASET = os.getenv("HUMAN_DATASET", "datasets/1-short.csv")
NO_HUMAN_DATASET = os.getenv("NO_HUMAN_DATASET", "datasets/2-short.csv")

BASIC_FEATURE_COLS = [
    "ls_re_mean",
    "ls_re_std",
    "ls_im_mean",
    "ls_im_std",
    "mag_mean",
    "mag_std",
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


class SPFRequest(BaseModel):
    frames: List[CSIFrame]


class HumanPresenceResult(BaseModel):
    timestamp: str
    sourceTimestamp: str
    sourceSensingUnit: str
    radioTac: str
    humanPresence: bool
    uncertaintyPercent: float
    modelId: str


class SPFResponse(BaseModel):
    results: List[HumanPresenceResult]


def _resolve_path(*candidates: str) -> str:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def _resolve_model_path() -> Optional[str]:
    candidates = [
        MODEL_PATH,
        "models/isac_human_presence.joblib",
        "models/human_presence_model.joblib",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _normalise_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "lsre" in df.columns and "ls_re" not in df.columns:
        rename_map["lsre"] = "ls_re"
    if "lsim" in df.columns and "ls_im" not in df.columns:
        rename_map["lsim"] = "ls_im"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"timestamp", "bin", "ls_re", "ls_im"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df[["timestamp", "bin", "ls_re", "ls_im"]].copy()


def _basic_stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_energy": 0.0,
        }

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=0)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_rms": float(np.sqrt(np.mean(arr ** 2))),
        f"{prefix}_energy": float(np.mean(arr ** 2)),
    }


def _features_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract both the original six simple features and the richer 76-feature set
    used by the packaged Random Forest model. This keeps the SPF compatible with
    old fallback models and the newer trained model bundle.
    """
    eps = 1e-9
    g = df.copy()
    re = g["ls_re"].astype(float).to_numpy()
    im = g["ls_im"].astype(float).to_numpy()
    bins = g["bin"].astype(float).to_numpy() if "bin" in g.columns else np.arange(len(g), dtype=float)

    mag = np.sqrt(re ** 2 + im ** 2)
    phase = np.angle(re + 1j * im)

    order = np.argsort(bins)
    re_s = re[order]
    im_s = im[order]
    mag_s = mag[order]

    dre = np.diff(re_s)
    dim = np.diff(im_s)
    dmag = np.diff(mag_s)

    features: Dict[str, float] = {
        # Backwards-compatible names used by the original SPF fallback model.
        "ls_re_mean": float(np.mean(re)),
        "ls_re_std": float(np.std(re, ddof=0)),
        "ls_im_mean": float(np.mean(im)),
        "ls_im_std": float(np.std(im, ddof=0)),
        "mag_mean": float(np.mean(mag)),
        "mag_std": float(np.std(mag, ddof=0)),
        # Rich-model feature names.
        "num_bins": float(len(g)),
    }

    features.update(_basic_stats("re", re))
    features.update(_basic_stats("im", im))
    features.update(_basic_stats("mag", mag))
    features.update(_basic_stats("phase", phase))
    features.update(_basic_stats("dre", dre))
    features.update(_basic_stats("dim", dim))
    features.update(_basic_stats("dmag", dmag))

    phase_cmean_re = float(np.mean(np.cos(phase)))
    phase_cmean_im = float(np.mean(np.sin(phase)))
    resultant_length = np.sqrt(phase_cmean_re ** 2 + phase_cmean_im ** 2)

    features["phase_cmean_re"] = phase_cmean_re
    features["phase_cmean_im"] = phase_cmean_im
    features["phase_cstd"] = float(np.sqrt(max(0.0, -2.0 * np.log(resultant_length + eps))))
    features["mag_cv"] = float(np.std(mag, ddof=0) / (np.mean(mag) + eps))

    if len(re) > 1 and np.std(re, ddof=0) > 0 and np.std(im, ddof=0) > 0:
        features["re_im_corr"] = float(np.corrcoef(re, im)[0, 1])
    else:
        features["re_im_corr"] = 0.0

    return pd.DataFrame([features])


def _build_training_features(df: pd.DataFrame, label: int) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("timestamp"):
        x = _features_from_frame(group)
        row = x.iloc[0].to_dict()
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


class HumanPresenceModel:
    def __init__(self) -> None:
        self.model_id = "fallback-random-forest"
        self.feature_cols = BASIC_FEATURE_COLS
        self.model = self._load_or_train_model()

    def _load_or_train_model(self):
        model_path = _resolve_model_path()
        if model_path is not None:
            bundle = joblib.load(model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.feature_cols = (
                    bundle.get("feature_columns")
                    or bundle.get("feature_cols")
                    or BASIC_FEATURE_COLS
                )
                self.model_id = bundle.get("model_id", Path(model_path).name)
                return bundle["model"]

            self.model_id = Path(model_path).name
            return bundle

        human_path = _resolve_path(HUMAN_DATASET, "datasets/1-short.csv", "1-short.csv")
        no_human_path = _resolve_path(NO_HUMAN_DATASET, "datasets/2-short.csv", "2-short.csv")
        df_human = _normalise_dataset_columns(pd.read_csv(human_path))
        df_no_human = _normalise_dataset_columns(pd.read_csv(no_human_path))

        train = pd.concat(
            [
                _build_training_features(df_human, label=1),
                _build_training_features(df_no_human, label=0),
            ],
            ignore_index=True,
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        clf.fit(train[self.feature_cols], train["label"])
        return clf

    def infer(self, frame: CSIFrame) -> tuple[bool, float]:
        df = pd.DataFrame(
            [{"bin": s.bin, "ls_re": s.ls_re, "ls_im": s.ls_im} for s in frame.samples]
        )
        features = _features_from_frame(df)

        for col in self.feature_cols:
            if col not in features.columns:
                features[col] = 0.0

        x = features[self.feature_cols]

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            # Expected class labels are 0=no-human, 1=human.
            classes = list(getattr(self.model, "classes_", [0, 1]))
            human_idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
            human_prob = float(proba[human_idx])
            max_prob = float(np.max(proba))
            return human_prob >= 0.5, (1.0 - max_prob) * 100.0

        prediction = int(self.model.predict(x)[0])
        return bool(prediction), 0.0


app = FastAPI(
    title="ISAC Sensing Processing Function",
    version="0.3.0",
    description="SPF running ML-based human presence detection on CSI frames.",
)

try:
    MODEL = HumanPresenceModel()
except Exception as exc:
    MODEL = None
    STARTUP_ERROR = str(exc)
else:
    STARTUP_ERROR = ""


@app.post("/process-csi", response_model=SPFResponse)
def process_csi(req: SPFRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail=STARTUP_ERROR)

    results: List[HumanPresenceResult] = []
    for frame in req.frames:
        human_present, uncertainty = MODEL.infer(frame)
        results.append(
            HumanPresenceResult(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                sourceTimestamp=frame.timestamp,
                sourceSensingUnit=frame.suId,
                radioTac=frame.radioTac,
                humanPresence=human_present,
                uncertaintyPercent=round(float(uncertainty), 3),
                modelId=MODEL.model_id,
            )
        )
    return SPFResponse(results=results)


@app.get("/healthz")
def healthcheck():
    return {
        "status": "ok" if MODEL is not None else "error",
        "modelId": MODEL.model_id if MODEL else None,
        "featureCount": len(MODEL.feature_cols) if MODEL else None,
        "modelPath": _resolve_model_path(),
        "startupError": STARTUP_ERROR,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("spf_service:app", host="0.0.0.0", port=PORT, reload=False)
