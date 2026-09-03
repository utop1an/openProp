from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.vlm_replay import read_captured_vlm_response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a captured VLM response against its exact safe input."
    )
    parser.add_argument("response", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    payload = read_captured_vlm_response(args.response, input_artifact=args.input)
    print(
        json.dumps(
            {
                "provider": payload["provider"],
                "model": payload["model"],
                "system_id": payload["system_id"],
                "input_episode_id": payload["input_episode_id"],
                "input_artifact_sha256": payload["input_artifact_sha256"],
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
