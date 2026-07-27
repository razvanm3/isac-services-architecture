"""
sensing_client.py

External Sensing Client for the CAMARA-style ISAC Exposure Function.

The client authenticates first through /oauth2/token, then invokes the protected
human-presence sensing API. It supports:
  1. one-shot sensing retrieval;
  2. continuous tracking/sensing mode, where the API is polled periodically and
     the complete observation history is stored as a JSON file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


TOKEN_ENDPOINT = "/oauth2/token"
HUMAN_PRESENCE_ENDPOINT = "/isac-human-presence/v0.1/retrieve"


@dataclass
class TokenSession:
    access_token: str
    token_type: str
    expires_at_epoch: float
    scope: str

    def is_expiring(self, margin_seconds: int = 30) -> bool:
        return time.time() >= self.expires_at_epoch - margin_seconds


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def authenticate(base_url: str, client_id: str, client_secret: str, scope: str) -> TokenSession:
    """Authenticate the client using a local OAuth2-like client-credentials flow."""
    url = base_url.rstrip("/") + TOKEN_ENDPOINT
    payload = {
        "grantType": "client_credentials",
        "clientId": client_id,
        "clientSecret": client_secret,
        "scope": scope,
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    expires_in = int(data.get("expiresIn", 3600))
    return TokenSession(
        access_token=data["accessToken"],
        token_type=data.get("tokenType", "Bearer"),
        expires_at_epoch=time.time() + expires_in,
        scope=data.get("scope", scope),
    )


def retrieve_human_presence(base_url: str, token: str, radio_tac: str, num_samples: int) -> Dict[str, Any]:
    """Retrieve a human-presence sensing result from the protected Exposure API."""
    url = base_url.rstrip("/") + HUMAN_PRESENCE_ENDPOINT
    payload = {
        "radioTac": radio_tac,
        "numSamples": num_samples,
    }
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(url, json=payload, headers=headers, timeout=40)
    resp.raise_for_status()
    return resp.json()


def save_json(data: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def summarize_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact terminal-friendly summary."""
    return {
        "requestId": result.get("requestId"),
        "clientId": result.get("clientId"),
        "radioTac": result.get("radioTac"),
        "humanPresence": result.get("humanPresence"),
        "confidencePercent": result.get("confidencePercent"),
        "averageUncertaintyPercent": result.get("averageUncertaintyPercent"),
        "topology": result.get("topology"),
        "sensingUnits": result.get("sensingUnits"),
    }


def run_one_shot(args: argparse.Namespace) -> None:
    token_session = authenticate(args.base_url, args.client_id, args.client_secret, args.scope)
    result = retrieve_human_presence(
        args.base_url,
        token_session.access_token,
        args.radio_tac,
        args.num_samples,
    )

    output = {
        "mode": "one-shot",
        "clientId": args.client_id,
        "radioTac": args.radio_tac,
        "requestedSamplesPerSU": args.num_samples,
        "retrievedAt": utc_now_iso(),
        "response": result,
    }
    save_json(output if args.wrap_output else result, args.output)

    print(f"Saved ISAC sensing response to {args.output}")
    print(json.dumps(summarize_response(result), indent=2))


def run_continuous(args: argparse.Namespace) -> None:
    """
    Continuous tracking/sensing mode.

    The client authenticates once, refreshes the token automatically before
    expiry, periodically calls the sensing endpoint, and persists the complete
    observation history to the configured JSON file after each iteration.
    """
    token_session = authenticate(args.base_url, args.client_id, args.client_secret, args.scope)

    started_at = utc_now_iso()
    started_epoch = time.time()
    observations: List[Dict[str, Any]] = []

    output: Dict[str, Any] = {
        "mode": "continuous",
        "clientId": args.client_id,
        "radioTac": args.radio_tac,
        "scope": args.scope,
        "startedAt": started_at,
        "finishedAt": None,
        "intervalSeconds": args.interval_seconds,
        "durationSeconds": args.duration_seconds,
        "maxIterations": args.max_iterations,
        "requestedSamplesPerSU": args.num_samples,
        "observations": observations,
    }

    iteration = 0
    print(
        "Starting continuous ISAC sensing "
        f"(radioTac={args.radio_tac}, interval={args.interval_seconds}s). "
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            if args.duration_seconds is not None and time.time() - started_epoch >= args.duration_seconds:
                break
            if args.max_iterations is not None and iteration >= args.max_iterations:
                break

            if token_session.is_expiring():
                token_session = authenticate(args.base_url, args.client_id, args.client_secret, args.scope)

            iteration += 1
            retrieved_at = utc_now_iso()

            try:
                result = retrieve_human_presence(
                    args.base_url,
                    token_session.access_token,
                    args.radio_tac,
                    args.num_samples,
                )
                observation = {
                    "iteration": iteration,
                    "retrievedAt": retrieved_at,
                    "ok": True,
                    "summary": summarize_response(result),
                    "response": result,
                }
                print(
                    f"[{iteration}] humanPresence={result.get('humanPresence')} "
                    f"confidence={result.get('confidencePercent')} "
                    f"uncertainty={result.get('averageUncertaintyPercent')} "
                    f"requestId={result.get('requestId')}"
                )
            except Exception as exc:
                observation = {
                    "iteration": iteration,
                    "retrievedAt": retrieved_at,
                    "ok": False,
                    "error": str(exc),
                }
                print(f"[{iteration}] sensing request failed: {exc}")

            observations.append(observation)
            output["finishedAt"] = utc_now_iso()
            save_json(output, args.output)

            if args.duration_seconds is not None:
                remaining = args.duration_seconds - (time.time() - started_epoch)
                if remaining <= 0:
                    break
                time.sleep(min(args.interval_seconds, remaining))
            else:
                time.sleep(args.interval_seconds)

    except KeyboardInterrupt:
        print("Continuous sensing stopped by user.")
    finally:
        output["finishedAt"] = utc_now_iso()
        save_json(output, args.output)
        print(f"Saved {len(observations)} continuous sensing observations to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMARA-style ISAC Sensing Client")
    parser.add_argument("--base-url", default="http://localhost:8500", help="Exposure Function base URL")
    parser.add_argument("--client-id", default="client-A", help="OAuth2 client identifier")
    parser.add_argument("--client-secret", default="client-A-secret", help="OAuth2 client secret")
    parser.add_argument("--scope", default="isac-human-presence:read", help="Requested OAuth2 scope")
    parser.add_argument("--radio-tac", default="226010001", help="Radio TAC for the sensing request")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of CSI frames per SU per request")
    parser.add_argument("--output", default="sensing_result.json", help="JSON output file")

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous tracking/sensing mode instead of a one-shot request.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=5.0,
        help="Polling interval in seconds for continuous mode.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional total duration in seconds for continuous mode.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional maximum number of sensing iterations for continuous mode.",
    )
    parser.add_argument(
        "--wrap-output",
        action="store_true",
        help="For one-shot mode, wrap the sensing response with client/run metadata.",
    )
    args = parser.parse_args()

    if args.continuous:
        run_continuous(args)
    else:
        run_one_shot(args)


if __name__ == "__main__":
    main()
