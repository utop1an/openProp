from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from openprop.component_balanced_grounding import aggregate_component_balanced_runs
from openprop.typed_context_ablation import aggregate_typed_context_runs


SUMMARY_KEYS = {
    "calibration": (
        "seeds",
        "aggregate",
        "paired_full_advantage",
        "simultaneous_primary_component_inference",
    ),
    "decision": (
        "seeds",
        "aggregate",
        "paired_probe_advantage",
        "simultaneous_probe_inference",
    ),
}


def refreshed_payload(
    payload: Mapping[str, Any],
    *,
    kind: str,
    aggregate: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    protocol = payload.get("protocol")
    runs = payload.get("runs")
    if not isinstance(protocol, Mapping) or not isinstance(runs, Iterable):
        raise ValueError("artifact must contain protocol and frozen runs")
    bootstrap_samples = int(protocol.get("bootstrap_samples", 0))
    if bootstrap_samples <= 0:
        raise ValueError("artifact must declare positive bootstrap_samples")
    summary = aggregate(tuple(runs), bootstrap_samples=bootstrap_samples)
    refreshed = dict(payload)
    for key in SUMMARY_KEYS[kind]:
        refreshed[key] = summary[key]
    refreshed_protocol = dict(protocol)
    refreshed_protocol["simultaneous_inference"] = (
        "paired bootstrap max standardized mean deviation; one shared seed "
        "resample across the three predeclared primary comparisons"
    )
    refreshed["protocol"] = refreshed_protocol
    return refreshed


def process(path: Path, *, kind: str, check: bool) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = (
        aggregate_typed_context_runs
        if kind == "calibration"
        else aggregate_component_balanced_runs
    )
    refreshed = refreshed_payload(payload, kind=kind, aggregate=aggregate)
    rendered = json.dumps(refreshed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if check:
        if path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"simultaneous inference drifted: {path}")
        print(f"verified: {path}")
        return
    path.write_text(rendered, encoding="utf-8")
    print(f"updated: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh multiplicity-aware component intervals from frozen runs"
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=Path("artifacts/typed_context_component_ablation.json"),
    )
    parser.add_argument(
        "--decision-artifact",
        type=Path,
        default=Path("artifacts/component_balanced_grounding_confirmation.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    process(args.calibration_artifact, kind="calibration", check=args.check)
    process(args.decision_artifact, kind="decision", check=args.check)


if __name__ == "__main__":
    main()
