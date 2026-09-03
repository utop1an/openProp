from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paper_claims import verify_claim_manifest


SCHEMA_VERSION = 2
MANIFEST_RELATIVE_PATH = "paper/reproducibility_manifest.json"

_SNAPSHOT_ROOT_FILES = ("AGENTS.md", "README.md", "pyproject.toml")
_SNAPSHOT_DIRECTORIES = ("src", "scripts", "tests", "examples", "docs", "paper")
_IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}

_EXPERIMENTS = (
    {
        "claim_ids": ["C1_TYPED_COMPOSITION", "N1_NEURAL_NECESSITY"],
        "entrypoint": "scripts/evaluate_compositional_multiseed.py",
        "command": "python scripts/evaluate_compositional_multiseed.py",
        "outputs": ["artifacts/compositional_persistence_multiseed_results.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C2_TYPED_COMPONENTS"],
        "entrypoint": "scripts/evaluate_typed_context_ablation.py",
        "command": "python scripts/evaluate_typed_context_ablation.py",
        "outputs": ["artifacts/typed_context_component_ablation.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C3_DECISION_UTILITY"],
        "entrypoint": "scripts/evaluate_component_balanced_grounding.py",
        "command": "python scripts/evaluate_component_balanced_grounding.py",
        "outputs": ["artifacts/component_balanced_grounding_confirmation.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C4_INTERVAL_CENSORING"],
        "entrypoint": "scripts/evaluate_observation_process.py",
        "command": "python scripts/evaluate_observation_process.py",
        "outputs": ["artifacts/observation_process_results.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C4_INTERVAL_CENSORING"],
        "entrypoint": "scripts/evaluate_observation_grounding.py",
        "command": "python scripts/evaluate_observation_grounding.py",
        "outputs": ["artifacts/observation_grounding_confirmation.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C6_FALSE_POSITIVE_OBSERVATIONS"],
        "entrypoint": "scripts/evaluate_false_positive_observation.py",
        "command": "python scripts/evaluate_false_positive_observation.py",
        "outputs": ["artifacts/false_positive_observation_results.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C7_RECURRENT_OBSERVATIONS"],
        "entrypoint": "scripts/evaluate_recurrent_observation.py",
        "command": "python scripts/evaluate_recurrent_observation.py",
        "outputs": ["artifacts/recurrent_observation_results.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C8_IRREGULAR_OBSERVATIONS"],
        "entrypoint": "scripts/evaluate_irregular_observation.py",
        "command": "python scripts/evaluate_irregular_observation.py",
        "outputs": ["artifacts/irregular_observation_results.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C9_SOURCE_RELIABILITY"],
        "entrypoint": "scripts/evaluate_source_reliability.py",
        "command": "python scripts/evaluate_source_reliability.py",
        "outputs": ["artifacts/source_reliability_results.json"],
        "requires_external_data": False,
    },
    {
        "claim_ids": ["C5_EXTERNAL_LANGUAGE"],
        "entrypoint": "scripts/evaluate_alfred_retrieval_baseline.py",
        "command": "python scripts/evaluate_alfred_retrieval_baseline.py --alfred-root <ALFRED_ROOT>",
        "outputs": [
            "artifacts/alfred_retrieval_baseline.json",
            "artifacts/alfred_retrieval_comparison.json",
            "artifacts/alfred_retrieval_vs_llm.json",
        ],
        "requires_external_data": True,
        "external_data_note": (
            "Official ALFRED train and validation trajectories are not vendored; "
            "follow docs/alfred-retrieval-baseline.md for the frozen multi-stage analysis."
        ),
    },
    {
        "claim_ids": ["N3_GENERAL_ADAPTATION_SAFETY"],
        "entrypoint": "scripts/evaluate_repeated_evidence_adaptation.py",
        "command": "python scripts/evaluate_repeated_evidence_adaptation.py",
        "outputs": ["artifacts/repeated_evidence_adaptation_development.json"],
        "requires_external_data": False,
        "stage": "development_only",
    },
)

_EXTERNAL_AUDITS = (
    {
        "entrypoint": "scripts/audit_teach_access.py",
        "command": "python scripts/audit_teach_access.py",
        "output": "artifacts/teach_access_audit.json",
        "requires_network": True,
        "claim_scope": (
            "official data-access and provenance audit; not dataset or model evidence"
        ),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"repository path must be nonempty and relative: {relative!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"repository path escapes root: {relative!r}") from error
    return candidate


def _claim_artifacts(claims_payload: Mapping[str, Any]) -> tuple[str, ...]:
    artifacts: set[str] = set()
    for claim in claims_payload.get("claims", []):
        for evidence in claim.get("evidence", []):
            relative = str(evidence.get("artifact", ""))
            if relative:
                artifacts.add(relative.replace("\\", "/"))
    return tuple(sorted(artifacts))


def _discover_snapshot_files(
    root: Path, claims_payload: Mapping[str, Any]
) -> tuple[str, ...]:
    files: set[str] = set()
    for relative in _SNAPSHOT_ROOT_FILES:
        if _safe_path(root, relative).is_file():
            files.add(relative)
    for directory in _SNAPSHOT_DIRECTORIES:
        base = _safe_path(root, directory)
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_RELATIVE_PATH:
                continue
            if any(part in _IGNORED_PARTS for part in path.parts):
                continue
            if path.suffix.casefold() in _IGNORED_SUFFIXES:
                continue
            files.add(relative)
    for relative in _claim_artifacts(claims_payload):
        if not _safe_path(root, relative).is_file():
            raise ValueError(f"claim artifact is missing from snapshot: {relative}")
        files.add(relative)
    return tuple(sorted(files))


def _snapshot_rows(root: Path, relatives: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in relatives:
        path = _safe_path(root, relative)
        if not path.is_file():
            raise ValueError(f"snapshot file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256(path)})
    return rows


def _tree_hash(rows: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(
        list(rows), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_revision(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if commit.returncode != 0:
            return {
                "commit": None,
                "dirty": None,
                "status": "unavailable",
                "reason": "git metadata unavailable",
            }
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if status.returncode != 0:
            return {
                "commit": commit.stdout.strip(),
                "dirty": None,
                "status": "partial",
                "reason": "git worktree status unavailable",
            }
        dirty = bool(status.stdout.strip())
        return {
            "commit": commit.stdout.strip(),
            "dirty": dirty,
            "status": "dirty" if dirty else "clean",
            "reason": None,
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return {
            "commit": None,
            "dirty": None,
            "status": "unavailable",
            "reason": "git metadata unavailable",
        }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(root: Path) -> dict[str, Any]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = project["project"]
    optional = metadata.get("optional-dependencies", {})
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "project": {
            "name": metadata["name"],
            "version": metadata["version"],
            "requires_python": metadata["requires-python"],
        },
        "declared_dependencies": {
            "core": list(metadata.get("dependencies", [])),
            **{name: list(values) for name, values in sorted(optional.items())},
        },
        "observed_distributions": {
            name: _package_version(name)
            for name in ("openprop", "torch", "matplotlib", "openai")
        },
        "torch_runtime": _torch_runtime(),
    }


def _torch_runtime() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    return {
        "version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def _experiment_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specification in _EXPERIMENTS:
        row = dict(specification)
        entrypoint = _safe_path(root, row["entrypoint"])
        if not entrypoint.is_file():
            raise ValueError(f"experiment entrypoint is missing: {row['entrypoint']}")
        outputs: list[dict[str, str]] = []
        for relative in row.pop("outputs"):
            output = _safe_path(root, relative)
            if not output.is_file():
                raise ValueError(f"experiment output is missing: {relative}")
            outputs.append({"path": relative, "sha256": _sha256(output)})
        row["outputs"] = outputs
        rows.append(row)
    return rows

def _external_audit_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specification in _EXTERNAL_AUDITS:
        row = dict(specification)
        entrypoint = _safe_path(root, row["entrypoint"])
        if not entrypoint.is_file():
            raise ValueError(f"external audit entrypoint is missing: {row['entrypoint']}")
        relative = str(row.pop("output"))
        output = _safe_path(root, relative)
        if not output.is_file():
            raise ValueError(f"external audit output is missing: {relative}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"external audit output must be an object: {relative}")
        if payload.get("claim_scope") != row["claim_scope"]:
            raise ValueError(f"external audit claim scope drifted: {relative}")
        decision = payload.get("decision")
        if not isinstance(decision, Mapping):
            raise ValueError(f"external audit decision is missing: {relative}")
        forbidden_true = (
            "layer_a_evidence_available",
            "layer_b_evidence_available",
            "layer_c_evidence_available",
            "performance_evidence",
        )
        if any(decision.get(name) is not False for name in forbidden_true):
            raise ValueError(f"external access audit cannot supply evidence: {relative}")
        row["output"] = {"path": relative, "sha256": _sha256(output)}
        row["observed_at_utc"] = payload.get("observed_at_utc")
        row["status"] = decision.get("status")
        rows.append(row)
    return rows


def build_reproducibility_manifest(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    claims_path = root / "paper" / "claims.json"
    claims_payload = json.loads(claims_path.read_text(encoding="utf-8"))
    claim_report = verify_claim_manifest(claims_path, repository_root=root)
    snapshot_files = _discover_snapshot_files(root, claims_payload)
    snapshot = _snapshot_rows(root, snapshot_files)
    revision = _git_revision(root)
    revision_bound = bool(revision["commit"]) and revision["dirty"] is False
    external_audits = _external_audit_rows(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Machine-verifiable computational snapshot for the evidence-locked "
            "OpenProp paper; external datasets are not redistributed."
        ),
        "revision": revision,
        "source_snapshot": {
            "algorithm": "sha256 over canonical sorted path/hash rows",
            "tree_sha256": _tree_hash(snapshot),
            "files": snapshot,
        },
        "environment": _environment(root),
        "install_profiles": {
            "core": 'python -m pip install -e "."',
            "deterministic_paper": 'python -m pip install -e ".[ml,paper]"',
            "external_language_collection": 'python -m pip install -e ".[openai]"',
        },
        "execution_contract": {
            "working_directory": "repository root",
            "environment": {"PYTHONPATH": "src"},
            "claim_verification": "python scripts/verify_paper_claims.py",
            "table_generation": "python scripts/build_paper_tables.py --check",
            "full_tests": "python -m unittest discover -s tests -v",
        },
        "experiments": _experiment_rows(root),
        "external_audits": external_audits,
        "external_inputs": {
            "alfred": {
                "vendored": False,
                "status": "required only for fresh external-language evaluation",
                "protocol": "docs/alfred-retrieval-baseline.md",
            },
            "teach": {
                "vendored": False,
                "status": "official longitudinal result pending",
                "access_audit": {
                    "path": external_audits[0]["output"]["path"],
                    "sha256": external_audits[0]["output"]["sha256"],
                    "observed_at_utc": external_audits[0]["observed_at_utc"],
                    "status": external_audits[0]["status"],
                    "performance_evidence": False,
                },
                "protocol": "docs/teach-feasibility-gate.md",
            },
        },
        "claim_verification": {
            "verified": claim_report["verified"],
            "claims": claim_report["claims"],
            "artifacts": claim_report["artifacts"],
            "metric_checks": claim_report["metric_checks"],
        },
        "release_gates": {
            "source_snapshot_complete": True,
            "claim_artifacts_verified": claim_report["verified"],
            "clean_git_revision_bound": revision_bound,
            "official_teach_result_available": False,
            "submission_release_ready": False,
        },
    }


def write_reproducibility_manifest(
    repository_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = build_reproducibility_manifest(repository_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def verify_reproducibility_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    require_runtime_match: bool = False,
    require_release_revision: bool = False,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else manifest_file.parent.parent.resolve()
    )
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"reproducibility manifest requires schema_version={SCHEMA_VERSION}")
    claims_payload = json.loads((root / "paper" / "claims.json").read_text(encoding="utf-8"))
    expected_files = _discover_snapshot_files(root, claims_payload)
    source = payload.get("source_snapshot")
    if not isinstance(source, Mapping) or not isinstance(source.get("files"), list):
        raise ValueError("reproducibility manifest requires source_snapshot.files")
    rows = source["files"]
    observed_paths = [str(row.get("path", "")) for row in rows if isinstance(row, Mapping)]
    if observed_paths != list(expected_files):
        raise ValueError("reproducibility source file inventory drifted")
    actual_rows = _snapshot_rows(root, expected_files)
    if rows != actual_rows:
        raise ValueError("reproducibility source file hash drifted")
    actual_tree_hash = _tree_hash(actual_rows)
    if source.get("tree_sha256") != actual_tree_hash:
        raise ValueError("reproducibility source tree hash drifted")
    claim_report = verify_claim_manifest(root / "paper" / "claims.json", repository_root=root)
    recorded_claims = payload.get("claim_verification")
    expected_claims = {
        "verified": claim_report["verified"],
        "claims": claim_report["claims"],
        "artifacts": claim_report["artifacts"],
        "metric_checks": claim_report["metric_checks"],
    }
    if recorded_claims != expected_claims:
        raise ValueError("reproducibility claim-verification summary drifted")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("reproducibility manifest requires experiments")
    output_count = 0
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise ValueError("every reproducibility experiment must be an object")
        entrypoint = _safe_path(root, str(experiment.get("entrypoint", "")))
        if not entrypoint.is_file():
            raise ValueError(f"reproducibility entrypoint is missing: {entrypoint}")
        for output in experiment.get("outputs", []):
            if not isinstance(output, Mapping):
                raise ValueError("experiment outputs must be objects")
            path = _safe_path(root, str(output.get("path", "")))
            if _sha256(path) != output.get("sha256"):
                raise ValueError(f"reproducibility experiment output drifted: {path}")
            output_count += 1
    external_audits = payload.get("external_audits")
    expected_external_audits = _external_audit_rows(root)
    if external_audits != expected_external_audits:
        raise ValueError("reproducibility external audit drifted")
    if require_runtime_match and payload.get("environment") != _environment(root):
        raise ValueError("current runtime does not match the recorded environment")
    gates = payload.get("release_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("reproducibility manifest requires release_gates")
    if require_release_revision and not gates.get("clean_git_revision_bound"):
        raise ValueError("release verification requires a clean bound git revision")
    return {
        "verified": True,
        "source_files": len(actual_rows),
        "source_tree_sha256": actual_tree_hash,
        "experiments": len(experiments),
        "experiment_outputs": output_count,
        "external_audits": len(expected_external_audits),
        "external_audit_outputs": len(expected_external_audits),
        "runtime_matched": bool(require_runtime_match),
        "release_revision_required": bool(require_release_revision),
        "release_ready": bool(gates.get("submission_release_ready")),
    }

