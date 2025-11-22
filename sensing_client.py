# sensing_client.py
# External sensing client calling the Exposure Function and saving results to CSV.

import argparse
import csv
from typing import Any, Dict

import requests


def call_human_presence_api(
    base_url: str,
    client_id: str,
    area_id: str,
    num_samples: int,
    su_mode: int,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/isac/human-presence/v0.1/detect"
    payload = {
        "clientId": client_id,
        "areaId": area_id,
        "numSamples": num_samples,
        "suMode": su_mode,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def save_results_to_csv(response: Dict[str, Any], output_path: str) -> None:
    fieldnames = [
        "timestamp",
        "human_presence",
        "uncertainty_percent",
        "client_id",
        "area_id",
        "topology_switched",
        "current_topology",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in response.get("results", []):
            writer.writerow({
                "timestamp": r["timestamp"],
                "human_presence": r["humanPresence"],
                "uncertainty_percent": r["uncertaintyPercent"],
                "client_id": response["clientId"],
                "area_id": response["areaId"],
                "topology_switched": response["topologySwitched"],
                "current_topology": response["currentTopology"],
            })


def main():
    parser = argparse.ArgumentParser(description="ISAC Sensing Client")
    parser.add_argument("--base-url", default="http://localhost:8500",
                        help="Exposure Function base URL")
    parser.add_argument("--client-id", required=True,
                        help="Client identifier (e.g., client-A)")
    parser.add_argument("--area-id", required=True,
                        help="Logical sensing area identifier (e.g., room-101)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of CSI snapshots to request")
    parser.add_argument("--su-mode", type=int, default=3, choices=[1, 2, 3],
                        help="SU mode: 1=human, 2=no-human, 3=variable")
    parser.add_argument("--output", default="sensing_results.csv",
                        help="CSV output file")

    args = parser.parse_args()

    resp = call_human_presence_api(
        base_url=args.base_url,
        client_id=args.client_id,
        area_id=args.area_id,
        num_samples=args.num_samples,
        su_mode=args.su_mode,
    )

    save_results_to_csv(resp, args.output)
    print(f"Saved {len(resp.get('results', []))} rows to {args.output}")


if __name__ == "__main__":
    main()
