from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


OBJECT_ACTIONS = {
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "ToggleObject",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "CoolObject",
}
ORDINAL_MARKERS = ("another", "other", "second", "next", "remaining")


def object_type(object_id: str) -> str:
    return object_id.split("|", 1)[0].strip().lower()


def description_for(low_action: dict[str, Any], descriptions: list[str]) -> str:
    high_idx = low_action.get("high_idx")
    if isinstance(high_idx, int) and 0 <= high_idx < len(descriptions):
        return str(descriptions[high_idx]).strip()
    return ""


def entity_lineage(object_id: str) -> str:
    """Collapse AI2-THOR child IDs (for example sliced pieces) to one entity lineage."""
    parts = [part.strip().lower() for part in object_id.split("|")]
    if len(parts) < 4:
        return object_id.strip().lower()
    return "|".join(parts[:4])


def trajectory_rows(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("turk_annotations", {}).get("anns", [])
    descriptions = annotations[0].get("high_descs", []) if annotations else []
    initial_counts = Counter()
    for pose in payload.get("scene", {}).get("object_poses", []):
        name = str(pose.get("objectName", ""))
        if name:
            initial_counts[name.split("_", 1)[0].strip().lower()] += 1

    histories: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict[str, Any]] = []
    object_steps = 0
    initially_ambiguous = 0
    repeated_target = 0
    multi_history = 0
    for low_idx, low_action in enumerate(payload.get("plan", {}).get("low_actions", [])):
        api_action = low_action.get("api_action", {})
        action = str(api_action.get("action", ""))
        target_id = api_action.get("objectId")
        if action not in OBJECT_ACTIONS or not isinstance(target_id, str) or not target_id:
            continue
        object_steps += 1
        target_type = object_type(target_id)
        target_lineage = entity_lineage(target_id)
        if initial_counts[target_type] >= 2:
            initially_ambiguous += 1
        prior = histories[target_type]
        if target_lineage in prior:
            repeated_target += 1
        if target_lineage in prior and len(prior) >= 2:
            multi_history += 1
            last_seen = {candidate: steps[-1] for candidate, steps in prior.items()}
            latest = max(last_seen.values())
            winners = sorted(candidate for candidate, step in last_seen.items() if step == latest)
            query = description_for(low_action, descriptions)
            rows.append(
                {
                    "split": split,
                    "task_id": str(payload.get("task_id", path.parent.name)),
                    "task_type": str(payload.get("task_type", "")),
                    "trajectory": path.as_posix(),
                    "low_action_index": low_idx,
                    "high_action_index": low_action.get("high_idx"),
                    "action": action,
                    "query": query,
                    "target_type": target_type,
                    "candidate_count": len(prior),
                    "target_last_seen_step": last_seen[target_lineage],
                    "latest_last_seen_step": latest,
                    "unique_recency_target": winners == [target_lineage],
                    "recency_tied": len(winners) > 1 and target_lineage in winners,
                    "recency_wrong": target_lineage not in winners,
                    "ordinal_language": any(marker in query.lower().split() for marker in ORDINAL_MARKERS),
                }
            )
        histories[target_type][target_lineage].append(low_idx)
    return rows, {
        "object_interaction_steps": object_steps,
        "initially_ambiguous_steps": initially_ambiguous,
        "repeated_target_steps": repeated_target,
        "multi_history_target_steps": multi_history,
    }


def iter_trajectories(root: Path, splits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for split in splits:
        split_root = root / split
        if not split_root.is_dir():
            raise ValueError(f"Missing ALFRED split: {split_root}")
        for path in sorted(split_root.rglob("traj_data.json")):
            yield split, path


def build_report(root: Path, splits: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = Counter()
    split_counts = Counter()
    trajectory_count = 0
    for split, path in iter_trajectories(root, splits):
        trajectory_count += 1
        split_counts[split] += 1
        trajectory_rows_result, counts = trajectory_rows(path, split)
        rows.extend(trajectory_rows_result)
        totals.update(counts)

    by_split = Counter(row["split"] for row in rows)
    by_action = Counter(row["action"] for row in rows)
    by_task_type = Counter(row["task_type"] for row in rows)
    recency = Counter(
        "target" if row["unique_recency_target"] else "tie" if row["recency_tied"] else "wrong"
        for row in rows
    )
    return {
        "schema_version": "alfred-longitudinal-feasibility-v1",
        "dataset_root": root.as_posix(),
        "splits": splits,
        "trajectory_count": trajectory_count,
        "trajectory_count_by_split": dict(sorted(split_counts.items())),
        "counts": dict(sorted(totals.items())),
        "eligible_multi_history_cases": len(rows),
        "eligible_by_split": dict(sorted(by_split.items())),
        "eligible_by_action": dict(sorted(by_action.items())),
        "eligible_by_task_type": dict(sorted(by_task_type.items())),
        "recency_outcomes": dict(sorted(recency.items())),
        "ordinal_language_cases": sum(bool(row["ordinal_language"]) for row in rows),
        "cases": rows,
        "claim_boundary": (
            "This audit measures whether official ALFRED plans contain repeated, same-type, "
            "multi-candidate object interactions that could support a history-only external "
            "grounding benchmark. It is not a grounding result and does not establish that "
            "action logs are equivalent to perceptual observation histories."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ALFRED for longitudinal grounding case feasibility")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["valid_seen", "valid_unseen"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.root.resolve(), args.splits)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
