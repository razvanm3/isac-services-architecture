# ISAC Services Architecture - v0.6 data-source-agnostic SU

This repository implements a service-based Integrated Sensing and Communications (ISAC) architecture for human-presence sensing. Version v0.6 simplifies the radio-log path: there is **no OAIBOX log service** and no legacy CSV simulator runtime path. The raw radio log is provided as a normal log file, and the Sensing Unit uses a helper module to transform that file into normalized CSI input samples.

The Sensing Unit does not call a simulator, a replay service, or any OAIBOX-specific API. It only receives a configured file path and imports the generic helper entry point:

```python
from sensing_unit_input_helper import parse_latest_csi_frames
```

The helper currently supports the OAIBOX/OAI SRS debug log structure, but this is hidden behind the helper interface. If another radio-log source is added later, it should be integrated inside the helper layer, not inside `su_service.py`.

---

## 1. Simplified architecture

Runtime service chain:

```text
Sensing Client
   │ 1. POST /oauth2/token
   │ 2. POST /isac-human-presence/v0.1/retrieve
   │    Authorization: Bearer <token>
   ▼
Exposure Function
   │ validates token, extracts client-id, checks allowed radio TACs
   ▼
Sensing Control Function (SeCF)
   │ orchestrates TAC-level sensing and topology decision
   ├───────────────► Sensing Processing Function (SPF)
   │                 applies the ML model and returns uncertainty
   ▼
Resource Allocation Function (RAF)
   │ discovers all SUs mapped to the requested radio TAC
   ▼
Sensing Unit(s)
   │ calls helper to obtain normalized CSI samples
   ▼
Sensing Unit Input Helper
   │ parses externally provided raw radio log file
   ▼
Raw radio log file
   │ created by real gNB/OAIBOX
```

For UAV-based SUs, the SU is also connected to the Power Transfer Orchestration Function (PTOF). If the UAV is within the activation radius of a passive sensor, human-presence sensing is suspended and the SU waveform is switched to `PTC-WAVEFORM-v1` for power transfer and communication.

```text
UAV-based SU ── GPS evaluation ──► PTOF ── dotted control ──► waveform / sensing state update
```

---

## 2. Repository structure

```text
sensing_unit_input_helper.py               # Helper: raw radio log -> normalized CSI frames
su_service.py                              # Data-source-agnostic SU service; supports UAV/PTOF mode
raf_service.py                             # Resource Allocation Function: TAC-based, multiple SUs per TAC
spf_service.py                             # Sensing Processing Function: loads trained ML model or fallback RF model
secf_service.py                            # Sensing Control Function: radioTac-based orchestration
exposure_service.py                        # CAMARA-style Exposure Function with token authentication
power_transfer_orchestration_service.py    # PTOF for UAV SU waveform switching
sensing_client.py                          # Client with one-shot and continuous sensing modes
evaluate_human_presence_model.py           # Offline model evaluation utility
isac_architecture_simplified.drawio        # Simplified draw.io architecture
requirements.txt                           # Python dependencies

samples/srs_debug_small.log                # Example raw SRS debug log
models/isac_human_presence.joblib          # Trained human-presence detection model
datasets/
1-short.csv                                # Reference CSI dataset used for ML evaluation/training only
2-short.csv                                # Reference CSI dataset used for ML evaluation/training only
```


---

## 3. Requirements

Use Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Expected dependencies:

```text
fastapi
uvicorn[standard]
pydantic
requests
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## 4. Raw radio log input model

The raw radio log file is created externally by the gNB/OAIBOX process. Each gNB power-up may create a new log file; the SU only needs the path of the currently active file.

The helper currently supports the OAIBOX/OAI SRS debug log format with:

- timestamp lines;
- `nr_srs_channel_estimation` function calls;
- UE port to gNB Rx antenna sections;
- RB-level SRS Tx/Rx/LS channel-estimation rows;
- interpolated channel/noise rows;
- signal power, per-RB noise/SNR and global noise/SNR metrics.

The helper converts this into the normalized CSI frame schema exposed by the SU:

```json
{
  "timestamp": "25:08:13:20:02:12:459990",
  "suId": "SU-1",
  "radioTac": "226010001",
  "samples": [
    {
      "bin": 0,
      "ls_re": 166.0,
      "ls_im": 392.0
    }
  ]
}
```

---

## 5. Running the architecture locally

Open separate terminals for each service.

### 5.1. Point to the raw log file

For a real OAIBOX/gNB test set the SU environment variable to the active log file path:

```bash
export SENSING_LOG_FILE=/path/to/current/srs_debug.log
```

### 5.2. Start the Power Transfer Orchestration Function

```bash
python power_transfer_orchestration_service.py
```

Default port: `8450`.

### 5.3. Start a terrestrial SU

```bash
SU_ID=SU-1 \
RADIO_TAC=226010001 \
SENSING_LOG_FILE=runtime/radio_logs/current_srs_debug.log \
CSI_LS_SCALE=1.0 \
IS_UAV_BASED=false \
PORT=8101 \
python su_service.py
```

Test the SU directly:

```bash
curl -X POST http://localhost:8101/csi \
  -H "Content-Type: application/json" \
  -d '{"numFrames": 3}'
```

### 5.4. Start a UAV-based SU

```bash
SU_ID=SU-2 \
RADIO_TAC=226010001 \
SENSING_LOG_FILE=runtime/radio_logs/current_srs_debug.log \
CSI_LS_SCALE=1.0 \
IS_UAV_BASED=true \
PTOF_BASE_URL=http://localhost:8450 \
PORT=8102 \
python su_service.py
```

The UAV-based SU periodically sends its GPS position to the PTOF. When it is near a passive sensor, human-presence sensing is suspended and the waveform is switched to the power-transfer/communication waveform.

### 5.5. Start RAF

```bash
python raf_service.py
```

Default port: `8200`.

### 5.6. Start SPF

```bash
MODEL_PATH=models/isac_human_presence.joblib python spf_service.py
```

Default port: `8300`.

### 5.7. Start SeCF

```bash
python secf_service.py
```

Default port: `8400`.

### 5.8. Start Exposure Function

```bash
python exposure_service.py
```

Default port: `8500`.

---

## 6. Sensing client usage

The sensing client authenticates first through:

```text
POST /oauth2/token
```

Then it invokes the protected sensing endpoint:

```text
POST /isac-human-presence/v0.1/retrieve
Authorization: Bearer <access-token>
```

### 6.1. One-shot sensing

```bash
python sensing_client.py \
  --base-url http://localhost:8500 \
  --client-id client-A \
  --client-secret client-A-secret \
  --radio-tac 226010001 \
  --num-samples 5 \
  --output sensing_result.json
```

The sensing API request body remains:

```json
{
  "radioTac": "226010001",
  "numSamples": 5
}
```

The client ID is not sent in the protected sensing request body. It is derived from the bearer token.

### 6.2. Continuous tracking/sensing mode

Continuous mode repeatedly retrieves sensing results at a configurable polling interval and stores the full observation history as JSON.

```bash
python sensing_client.py \
  --continuous \
  --base-url http://localhost:8500 \
  --client-id client-A \
  --client-secret client-A-secret \
  --radio-tac 226010001 \
  --num-samples 3 \
  --interval-seconds 2 \
  --max-iterations 10 \
  --output continuous_sensing_result.json
```

The client refreshes the bearer token automatically before token expiry.

---

## 7. ML model and evaluation

The SPF expects a trained model at:

```text
models/isac_human_presence.joblib
```

Offline model evaluation:

```bash
python evaluate_human_presence_model.py \
  --model-path models/isac_human_presence.joblib \
  --human-csv 1-short.csv \
  --no-human-csv 2-short.csv \
  --output-dir model_evaluation
```

Important validation note: the current model was trained on normalized CSI features with columns:

```text
timestamp, bin, ls_re, ls_im
```

The helper produces the same minimum schema, so the SPF can process the SU output without API changes. However, raw gNB/OAIBOX LS values may have a different numerical scale than the initial CSV datasets. Use:

```text
CSI_LS_SCALE=<scale_factor>
```

or retrain the model on CSI data generated through the same helper path.

---

## 8. Main API endpoints


### Sensing Unit

```text
GET  /capabilities
GET  /state
POST /csi
POST /control/waveform
GET  /healthz
```

### Power Transfer Orchestration Function

```text
POST /power-transfer/evaluate
GET  /passive-sensors
GET  /uav-sus/{su_id}/decision
GET  /healthz
```

### RAF

```text
GET  /capabilities
POST /measurements
GET  /healthz
```

### SPF

```text
POST /process-csi
GET  /healthz
```

### SeCF

```text
POST /sensing-requests
GET  /healthz
```

### Exposure Function

```text
POST /oauth2/token
POST /isac-human-presence/v0.1/retrieve
GET  /healthz
```

---

## 9. Authentication model

The Exposure Function currently implements a local OAuth2-like client-credentials flow. Demo client profiles are defined in `exposure_service.py`:

```python
CLIENT_AUTH_PROFILES = {
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
```

The issued bearer token embeds:

```text
clientId
allowedRadioTacs
scope
iat
exp
```

The sensing request body does not contain `clientId`; the Exposure Function derives the client identity from the token.

---

