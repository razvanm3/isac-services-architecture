"""
sensing_unit_input_helper.py

Helper functions used by the Sensing Unit (SU) to transform externally provided
raw radio logs into normalized CSI input samples.

The SU service imports only the generic function parse_latest_csi_frames().
Therefore, the SU runtime does not need to know whether the raw log was produced
by OAIBOX/OAI, a real gNB, or another capture process. The current parser
implementation supports the OAIBOX/OAI SRS debug log structure:

  Timestamp: yy:mm:dd:HH:MM:SS:us
  Calling nr_srs_channel_estimation function
  UE port <portNo> --> gNB Rx antenna <antennaNo>
  RB blocks containing gen/rx/ls rows and optional ls/interp/noise rows
  signal_power / per-RB noise / global noise/SNR metrics

The output schema is the normalized CSI frame format consumed by the RAF/SPF
chain:

  {
    "timestamp": "...",
    "suId": "SU-1",
    "radioTac": "226010001",
    "samples": [{"bin": 0, "ls_re": 166.0, "ls_im": 392.0}]
  }
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Regex patterns for the currently supported raw SRS debug log format.
# -----------------------------------------------------------------------------

TS_RE = re.compile(r"^Timestamp:\s+(.+?)\s*$")
CALL_RE = re.compile(r"^Calling\s+nr_srs_channel_estimation function")
UE_HDR_RE = re.compile(r"UE port\s+(\d+)\s*-->\s*gNB Rx antenna\s+(\d+)")
BLOCK_SEP_RE = re.compile(r"^-+\s+(\d+)\s+-+\s*$")
GRID1_HDR_RE = re.compile(r"__genRe")  # gen/rx/ls grid
GRID2_HDR_RE = re.compile(r"__lsRe")   # ls/int/noi grid
ROW6_RE = re.compile(
    r"^\(\s*(\d+)\)\s*"
    r"([+-]?\d+)\s+([+-]?\d+)\s*\|\s*"
    r"([+-]?\d+)\s+([+-]?\d+)\s*\|\s*"
    r"([+-]?\d+)\s+([+-]?\d+)\s*$"
)
SIGNAL_POWER_RE = re.compile(r"^signal_power\s*=\s*(\d+)\s*$")
NOISE_RB_RE = re.compile(
    r"^noise_power_per_rb\[(\d+)\]\s*=\s*(\d+),\s*"
    r"snr_per_rb\[(\d+)\]\s*=\s*(\d+)\s*dB\s*$"
)
NOISE_POWER_RE = re.compile(r"^noise_power\s*=\s*(\d+),\s*SNR\s*=\s*(\d+)\s*dB\s*$")


def timestamp_to_microseconds(ts: str) -> int:
    """Convert yy:mm:dd:HH:MM:SS:us to Unix epoch microseconds."""
    parts = ts.split(":")
    if len(parts) != 7:
        raise ValueError(f"Unexpected timestamp format: {ts}")
    year, month, day, hour, minute, second, microsec = map(int, parts)
    dt = datetime(year=2000 + year, month=month, day=day, hour=hour, minute=minute, second=second)
    return int(dt.timestamp() * 1_000_000) + microsec


class _RecordBuilder:
    def __init__(self, timestamp: Optional[str]):
        self.timestamp = timestamp
        self.sections: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.metrics: Dict[str, Any] = {
            "signal_power": None,
            "per_rb": [],
            "noise_power": None,
            "snr_db": None,
        }
        self.complete = False

    def ensure_section(self, port: int, antenna: int) -> Dict[str, Any]:
        key = (port, antenna)
        if key not in self.sections:
            self.sections[key] = {"port": port, "antenna": antenna, "rb": {}}
        return self.sections[key]

    @staticmethod
    def ensure_rb(section: Dict[str, Any], rb_index: int) -> Dict[str, Any]:
        rb_map = section["rb"]
        if rb_index not in rb_map:
            rb_map[rb_index] = {"rb_index": rb_index, "genrxls": [], "lsintnoi": []}
        return rb_map[rb_index]

    def to_jsonable(self) -> Dict[str, Any]:
        sections: List[Dict[str, Any]] = []
        for section in self.sections.values():
            rb_entries = list(section["rb"].values())
            rb_entries.sort(key=lambda x: x["rb_index"])
            sections.append({
                "port": section["port"],
                "antenna": section["antenna"],
                "rb": rb_entries,
            })
        sections.sort(key=lambda x: (x["port"], x["antenna"]))
        return {
            "timestamp": self.timestamp,
            "sections": sections,
            "metrics": self.metrics,
            "complete": self.complete,
        }


def parse_raw_srs_debug_log(
    input_path: str | os.PathLike[str],
    *,
    only_complete_records: bool = True,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Parse a raw SRS debug log into in-memory records.

    The parser tolerates partially written live logs. If
    only_complete_records=True, the last incomplete record is ignored.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Radio log file not found: {path}")

    records: List[Dict[str, Any]] = []
    last_ts: Optional[str] = None
    current: Optional[_RecordBuilder] = None
    current_port: Optional[int] = None
    current_ant: Optional[int] = None
    current_block: Optional[int] = None
    current_grid: Optional[str] = None

    def finish_record(force: bool = False) -> None:
        nonlocal current, current_port, current_ant, current_block, current_grid
        if current is not None:
            obj = current.to_jsonable()
            if force or not only_complete_records or obj.get("complete"):
                if obj.get("sections"):
                    records.append(obj)
        current = None
        current_port = None
        current_ant = None
        current_block = None
        current_grid = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            m = TS_RE.match(line)
            if m:
                last_ts = m.group(1).strip()
                continue

            if CALL_RE.match(line):
                finish_record(force=False)
                current = _RecordBuilder(timestamp=last_ts)
                continue

            m = UE_HDR_RE.search(line)
            if m:
                current_port = int(m.group(1))
                current_ant = int(m.group(2))
                if current is None:
                    current = _RecordBuilder(timestamp=last_ts)
                current.ensure_section(current_port, current_ant)
                continue

            if GRID1_HDR_RE.search(line):
                current_grid = "genrxls"
                continue

            if GRID2_HDR_RE.search(line):
                current_grid = "lsintnoi"
                continue

            m = BLOCK_SEP_RE.match(line)
            if m:
                current_block = int(m.group(1))
                continue

            m = ROW6_RE.match(line)
            if (
                m
                and current is not None
                and current_port is not None
                and current_ant is not None
                and current_block is not None
                and current_grid
            ):
                bin_idx = int(m.group(1))
                a1, a2, b1, b2, c1, c2 = map(int, m.groups()[1:])
                section = current.ensure_section(current_port, current_ant)
                rb = current.ensure_rb(section, current_block)
                if current_grid == "genrxls":
                    rb["genrxls"].append({
                        "bin": bin_idx,
                        "gen": {"re": a1, "im": a2},
                        "rx": {"re": b1, "im": b2},
                        "ls": {"re": c1, "im": c2},
                    })
                else:
                    rb["lsintnoi"].append({
                        "bin": bin_idx,
                        "ls": {"re": a1, "im": a2},
                        "interp": {"re": b1, "im": b2},
                        "noise": {"re": c1, "im": c2},
                    })
                continue

            if current is not None:
                m = SIGNAL_POWER_RE.match(line)
                if m:
                    current.metrics["signal_power"] = int(m.group(1))
                    continue

                m = NOISE_RB_RE.match(line)
                if m:
                    current.metrics["per_rb"].append({
                        "rb": int(m.group(1)),
                        "noise_power": int(m.group(2)),
                        "snr_db": int(m.group(4)),
                    })
                    continue

                m = NOISE_POWER_RE.match(line)
                if m:
                    current.metrics["noise_power"] = int(m.group(1))
                    current.metrics["snr_db"] = int(m.group(2))
                    current.complete = True
                    finish_record(force=True)
                    if max_records is not None and len(records) >= max_records:
                        break
                    continue

    # End-of-file. Do not force an incomplete live record unless requested.
    if current is not None:
        finish_record(force=not only_complete_records)

    if max_records is not None and len(records) > max_records:
        return records[-max_records:]
    return records


def records_to_csi_frames(
    records: Iterable[Dict[str, Any]],
    *,
    su_id: str,
    radio_tac: str,
    max_frames: Optional[int] = None,
    ls_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Convert parsed records into normalized CSI frames used by the RAF/SPF chain.

    The LS channel estimates are extracted from the gen/rx/ls grid. If a record
    contains only the ls/interp/noise grid, the function falls back to that LS
    field.
    """
    selected = list(records)
    if max_frames is not None:
        selected = selected[-max_frames:]

    frames: List[Dict[str, Any]] = []

    for rec in selected:
        samples: List[Dict[str, Any]] = []
        for section in rec.get("sections", []):
            for rb in section.get("rb", []):
                source_rows = rb.get("genrxls") or rb.get("lsintnoi") or []
                for row in source_rows:
                    ls = row.get("ls", {})
                    if "re" not in ls or "im" not in ls:
                        continue
                    samples.append({
                        "bin": int(row["bin"]),
                        "ls_re": float(ls["re"]) / ls_scale,
                        "ls_im": float(ls["im"]) / ls_scale,
                    })

        if samples:
            frames.append({
                "timestamp": rec.get("timestamp") or "unknown",
                "suId": su_id,
                "radioTac": radio_tac,
                "samples": samples,
            })

    return frames


def parse_latest_csi_frames(
    log_path: str | os.PathLike[str],
    *,
    su_id: str,
    radio_tac: str,
    num_frames: int,
    ls_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Generic entry point used by the SU service.

    The SU service should call this function and treat the returned objects as
    input samples, without caring about the underlying log source format.
    """
    records = parse_raw_srs_debug_log(log_path, only_complete_records=True)
    return records_to_csi_frames(
        records,
        su_id=su_id,
        radio_tac=radio_tac,
        max_frames=num_frames,
        ls_scale=ls_scale,
    )
