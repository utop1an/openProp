from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_GAME_SUFFIX = ".game.json"
_STATE_PATTERN = re.compile(r"^statediff\.(?P<time>.+)\.json$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class _PreparedTeachSession:
    split: str
    game_id: str
    episode_id: str
    floorplan: str
    game_file: Path
    state_directory: Path
    initial_state: Mapping[str, Any]
    final_timestamp: float
    numeric_state_files: tuple[Path, ...]
    final_state_file: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return payload


def _safe_component(value: Any, *, field: str) -> str:
    result = str(value).strip()
    if _SAFE_COMPONENT.fullmatch(result) is None:
        raise ValueError(f"TEACh {field} must be a safe nonempty path component: {result!r}")
    return result


def _single_episode(
    payload: Mapping[str, Any],
    *,
    game_file: Path,
) -> Mapping[str, Any]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"TEACh game {game_file} requires a nonempty tasks list")
    episodes: list[Mapping[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("episodes"), list):
            raise ValueError(f"TEACh game {game_file} has an invalid task episode list")
        for episode in task["episodes"]:
            if not isinstance(episode, Mapping):
                raise ValueError(f"TEACh game {game_file} contains a non-object episode")
            episodes.append(episode)
    if len(episodes) != 1:
        raise ValueError(
            f"TEACh replay directory is game-level, so {game_file} must contain "
            f"exactly one episode; found {len(episodes)}"
        )
    return episodes[0]


def _interaction_times(
    episode: Mapping[str, Any],
    *,
    game_file: Path,
) -> tuple[tuple[float, ...], float]:
    interactions = episode.get("interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError(f"TEACh episode in {game_file} requires interactions")
    starts: list[float] = []
    ends: list[float] = []
    previous = -math.inf
    for index, interaction in enumerate(interactions):
        if not isinstance(interaction, Mapping):
            raise ValueError(f"TEACh interaction {index} in {game_file} must be an object")
        try:
            start = float(interaction["time_start"])
            duration = float(interaction["duration"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid TEACh interaction timestamp at index {index} in {game_file}: {error}"
            ) from error
        if (
            not math.isfinite(start)
            or not math.isfinite(duration)
            or start < 0.0
            or duration < 0.0
            or start < previous
        ):
            raise ValueError(
                f"TEACh interaction times in {game_file} must be finite, nonnegative, and ordered"
            )
        previous = start
        starts.append(start)
        ends.append(start + duration)
    if len(starts) != len(set(starts)):
        raise ValueError(f"TEACh interaction times in {game_file} must be unique")
    final_timestamp = max(ends)
    if not math.isfinite(final_timestamp):
        raise ValueError(f"TEACh final timestamp in {game_file} is nonfinite")
    return tuple(starts), final_timestamp


def _state_files(
    state_directory: Path,
    *,
    expected_times: tuple[float, ...],
) -> tuple[tuple[Path, ...], Path]:
    if not state_directory.is_dir():
        raise ValueError(f"missing TEACh replay state directory: {state_directory}")
    numeric: list[tuple[float, Path]] = []
    final: list[Path] = []
    for path in state_directory.glob("statediff.*.json"):
        match = _STATE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        token = match.group("time")
        _json_object(path, label="TEACh state diff")
        if token == "end":
            final.append(path)
            continue
        try:
            timestamp = float(token)
        except ValueError as error:
            raise ValueError(f"invalid TEACh state timestamp in {path}") from error
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"invalid TEACh state timestamp in {path}")
        numeric.append((timestamp, path))
    if len(final) != 1:
        raise ValueError(
            f"TEACh replay {state_directory} requires exactly one statediff.end.json"
        )
    observed_times = tuple(value for value, _ in sorted(numeric))
    if observed_times != tuple(sorted(expected_times)):
        raise ValueError(
            f"TEACh replay timestamps do not match game interactions in {state_directory}: "
            f"expected {len(expected_times)}, observed {len(observed_times)}"
        )
    return tuple(path for _, path in sorted(numeric)), final[0]


def _relative_path(path: Path, parent: Path) -> str:
    try:
        return Path(os.path.relpath(path, parent)).as_posix()
    except ValueError:
        return str(path.resolve())


def _state_inventory_sha256(
    numeric: tuple[Path, ...],
    final: Path,
) -> str:
    digest = hashlib.sha256()
    for path in (*numeric, final):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _discover_splits(
    games_root: Path,
    states_root: Path,
    requested: Iterable[str] | None,
) -> tuple[str, ...]:
    if not games_root.is_dir():
        raise ValueError(f"missing official TEACh games directory: {games_root}")
    if not states_root.is_dir():
        raise ValueError(f"missing official TEACh images/states directory: {states_root}")
    available_games = {path.name for path in games_root.iterdir() if path.is_dir()}
    available_states = {path.name for path in states_root.iterdir() if path.is_dir()}
    if requested is None:
        splits = tuple(sorted(available_games & available_states))
    else:
        splits = tuple(sorted({_safe_component(value, field="split") for value in requested}))
    if not splits:
        raise ValueError("no common TEACh game/state splits were found")
    missing_games = set(splits) - available_games
    missing_states = set(splits) - available_states
    if missing_games or missing_states:
        raise ValueError(
            "requested TEACh splits are incomplete: "
            f"missing games={sorted(missing_games)}, missing states={sorted(missing_states)}"
        )
    return splits


def prepare_official_teach_manifest(
    *,
    games_root: str | Path,
    states_root: str | Path,
    output_manifest: str | Path,
    initial_state_directory: str | Path,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a strict OpenProp manifest from extracted official TEACh archives.

    Official replay state directories are keyed by game filename and contain
    state diffs relative to the episode initial state. This function validates
    that every selected game has one episode, one replay state per interaction,
    and one final diff before materializing initial states and the manifest.
    Missing or ambiguous sessions fail rather than being silently excluded.
    """

    games = Path(games_root).resolve()
    states = Path(states_root).resolve()
    manifest = Path(output_manifest).resolve()
    initial_root = Path(initial_state_directory).resolve()
    selected_splits = _discover_splits(games, states, splits)
    prepared: list[_PreparedTeachSession] = []
    seen_games: set[str] = set()
    seen_episodes: set[str] = set()
    games_per_split: dict[str, int] = {}
    for split in selected_splits:
        game_files = tuple(sorted((games / split).glob(f"*{_GAME_SUFFIX}")))
        if not game_files:
            raise ValueError(f"TEACh split {split!r} contains no game files")
        games_per_split[split] = len(game_files)
        for game_file in game_files:
            game_id = _safe_component(
                game_file.name[: -len(_GAME_SUFFIX)], field="game_id"
            )
            if game_id in seen_games:
                raise ValueError(f"duplicate TEACh game_id across splits: {game_id}")
            payload = _json_object(game_file, label="TEACh game")
            episode = _single_episode(payload, game_file=game_file)
            episode_id = _safe_component(episode.get("episode_id"), field="episode_id")
            if episode_id in seen_episodes:
                raise ValueError(f"duplicate TEACh episode_id across games: {episode_id}")
            floorplan = _safe_component(episode.get("world"), field="floorplan")
            initial_state = episode.get("initial_state")
            if not isinstance(initial_state, Mapping) or not isinstance(
                initial_state.get("objects"), list
            ):
                raise ValueError(
                    f"TEACh episode {episode_id!r} requires initial_state.objects"
                )
            expected_times, final_timestamp = _interaction_times(
                episode, game_file=game_file
            )
            state_directory = states / split / game_id
            numeric, final = _state_files(
                state_directory,
                expected_times=expected_times,
            )
            if final_timestamp < max(expected_times):
                raise ValueError(
                    f"TEACh final timestamp precedes replay observations for {game_id}"
                )
            seen_games.add(game_id)
            seen_episodes.add(episode_id)
            prepared.append(
                _PreparedTeachSession(
                    split=split,
                    game_id=game_id,
                    episode_id=episode_id,
                    floorplan=floorplan,
                    game_file=game_file.resolve(),
                    state_directory=state_directory.resolve(),
                    initial_state=dict(initial_state),
                    final_timestamp=final_timestamp,
                    numeric_state_files=numeric,
                    final_state_file=final,
                )
            )
    if not prepared:
        raise ValueError("official TEACh preparation produced no sessions")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    initial_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in prepared:
        initial_path = initial_root / item.split / f"{item.game_id}.initial.json"
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_bytes = (
            json.dumps(
                item.initial_state,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        initial_path.write_bytes(initial_bytes)
        rows.append(
            {
                "episode_id": item.episode_id,
                "floorplan": item.floorplan,
                "initial_state": _relative_path(initial_path, manifest.parent),
                "state_directory": _relative_path(
                    item.state_directory, manifest.parent
                ),
                "final_timestamp": item.final_timestamp,
                "game_file": _relative_path(item.game_file, manifest.parent),
                "official_split": item.split,
                "official_game_id": item.game_id,
                "game_file_sha256": _sha256(item.game_file),
                "initial_state_sha256": hashlib.sha256(initial_bytes).hexdigest(),
                "state_inventory_sha256": _state_inventory_sha256(
                    item.numeric_state_files,
                    item.final_state_file,
                ),
                "numeric_state_files": len(item.numeric_state_files),
            }
        )
    rows.sort(key=lambda row: (row["official_split"], row["official_game_id"]))
    manifest_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    temporary = manifest.with_name(manifest.name + ".tmp")
    temporary.write_bytes(manifest_bytes)
    temporary.replace(manifest)
    return {
        "protocol": {
            "source": "official TEACh all_games and images_and_states archives",
            "layout": "games/<split>/*.game.json paired with images/<split>/<game_id>",
            "selection": "all games in selected splits; missing or ambiguous replay fails",
            "initial_state": "materialized from the unique official game episode",
            "final_timestamp": "maximum interaction time_start plus duration",
            "claim_scope": "data preparation and integrity; not model performance",
        },
        "games_root": str(games),
        "states_root": str(states),
        "manifest": str(manifest),
        "initial_state_directory": str(initial_root),
        "splits": list(selected_splits),
        "games_per_split": games_per_split,
        "sessions": len(rows),
        "floorplans": len({row["floorplan"] for row in rows}),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "rows": rows,
    }


def write_teach_manifest_preparation_report(
    path: str | Path,
    report: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

