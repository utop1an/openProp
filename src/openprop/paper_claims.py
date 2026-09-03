from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_ALLOWED_STATUSES = {
    "supported_synthetic",
    "supported_external_language",
    "contradicted",
    "unsupported",
    "pending_external",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    current = payload
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as error:
                raise ValueError(f"invalid list token {token!r} in {pointer!r}") from error
        elif isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"missing object token {token!r} in {pointer!r}")
            current = current[token]
        else:
            raise ValueError(f"pointer {pointer!r} traverses a scalar at {token!r}")
    return current


def _safe_repository_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"claim evidence escapes repository root: {relative!r}") from error
    return candidate


def verify_claim_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify evidence hashes and exact metric bindings for paper claims."""

    manifest_file = Path(manifest_path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else manifest_file.parent.parent.resolve()
    )
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("paper claim manifest requires schema_version=1")
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("paper claim manifest requires a nonempty claims list")

    seen: set[str] = set()
    artifact_cache: dict[Path, Any] = {}
    checked_artifacts: set[Path] = set()
    check_count = 0
    status_counts: dict[str, int] = {}
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise ValueError("every paper claim must be an object")
        claim_id = str(claim.get("id", "")).strip()
        status = str(claim.get("status", "")).strip()
        if not claim_id or claim_id in seen:
            raise ValueError(f"paper claim ids must be nonempty and unique: {claim_id!r}")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"invalid paper claim status for {claim_id}: {status!r}")
        if not str(claim.get("claim", "")).strip() or not str(claim.get("scope", "")).strip():
            raise ValueError(f"paper claim {claim_id} requires claim and scope text")
        seen.add(claim_id)
        status_counts[status] = status_counts.get(status, 0) + 1
        evidence_rows = claim.get("evidence", [])
        if not isinstance(evidence_rows, list):
            raise ValueError(f"paper claim {claim_id} evidence must be a list")
        if status.startswith("supported_") and not evidence_rows:
            raise ValueError(f"supported paper claim {claim_id} requires evidence")
        for evidence in evidence_rows:
            if not isinstance(evidence, Mapping):
                raise ValueError(f"paper claim {claim_id} evidence must be an object")
            artifact = _safe_repository_path(root, str(evidence.get("artifact", "")))
            if not artifact.is_file():
                raise ValueError(f"paper claim {claim_id} artifact is missing: {artifact}")
            expected_hash = str(evidence.get("sha256", "")).casefold()
            actual_hash = _sha256(artifact)
            if len(expected_hash) != 64 or actual_hash != expected_hash:
                raise ValueError(
                    f"paper claim {claim_id} artifact hash mismatch for {artifact}: "
                    f"expected {expected_hash}, observed {actual_hash}"
                )
            checked_artifacts.add(artifact)
            if artifact not in artifact_cache:
                artifact_cache[artifact] = json.loads(artifact.read_text(encoding="utf-8"))
            checks = evidence.get("checks", [])
            if not isinstance(checks, list) or not checks:
                raise ValueError(f"paper claim {claim_id} evidence requires metric checks")
            for check in checks:
                if not isinstance(check, Mapping):
                    raise ValueError(f"paper claim {claim_id} metric check must be an object")
                pointer = str(check.get("pointer", ""))
                observed = _json_pointer(artifact_cache[artifact], pointer)
                expected = check.get("expected")
                tolerance = float(check.get("absolute_tolerance", 0.0))
                if tolerance < 0 or not math.isfinite(tolerance):
                    raise ValueError(f"paper claim {claim_id} has invalid tolerance")
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
                        raise ValueError(
                            f"paper claim {claim_id} expected numeric value at {pointer}"
                        )
                    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
                        raise ValueError(
                            f"paper claim {claim_id} metric mismatch at {pointer}: "
                            f"expected {expected} +/- {tolerance}, observed {observed}"
                        )
                elif observed != expected:
                    raise ValueError(
                        f"paper claim {claim_id} value mismatch at {pointer}: "
                        f"expected {expected!r}, observed {observed!r}"
                    )
                check_count += 1
    return {
        "manifest": str(manifest_file),
        "repository_root": str(root),
        "claims": len(claims),
        "status_counts": dict(sorted(status_counts.items())),
        "artifacts": len(checked_artifacts),
        "metric_checks": check_count,
        "verified": True,
    }
