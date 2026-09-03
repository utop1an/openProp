from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OFFICIAL_TEACH_REPOSITORY = "https://github.com/alexa/teach"
OFFICIAL_TEACH_COMMIT = "903191e256da866a603d1bbfb21db34e0874392d"
OFFICIAL_TEACH_DOWNLOADER_PATH = "src/teach/cli/download.py"
OFFICIAL_TEACH_ACCESS_ISSUE = "https://github.com/alexa/teach/issues/37"
REQUIRED_TEACH_ARCHIVES = (
    "all_games.tar.gz",
    "images_and_states.tar.gz",
)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MAINTAINER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_official_teach_downloader(source: str) -> dict[str, Any]:
    """Extract the bucket and archive inventory from the official downloader."""

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"invalid official TEACh downloader source: {error}") from error
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "DEFAULT_DATASET_BUCKET_NAME",
            "FILE_LIST",
        }:
            continue
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"official TEACh downloader {target.id} must be a literal"
            ) from error
    bucket = assignments.get("DEFAULT_DATASET_BUCKET_NAME")
    file_list = assignments.get("FILE_LIST")
    if not isinstance(bucket, str) or not bucket.strip():
        raise ValueError("official TEACh downloader is missing its dataset bucket")
    if not isinstance(file_list, list) or not all(
        isinstance(value, str) and value for value in file_list
    ):
        raise ValueError("official TEACh downloader is missing a literal file list")
    if len(file_list) != len(set(file_list)):
        raise ValueError("official TEACh downloader contains duplicate archive names")
    missing = set(REQUIRED_TEACH_ARCHIVES) - set(file_list)
    if missing:
        raise ValueError(
            f"official TEACh downloader omits required archives: {sorted(missing)}"
        )
    return {"bucket": bucket.strip(), "file_list": tuple(file_list)}


def official_archive_urls(bucket: str, archive: str) -> tuple[str, str]:
    if not bucket or "/" in bucket or not archive or "/" in archive:
        raise ValueError("TEACh bucket and archive must be simple path components")
    return (
        f"https://{bucket}.s3.amazonaws.com/{archive}",
        f"https://s3.amazonaws.com/{bucket}/{archive}",
    )


def inspect_local_teach_archives(
    directory: str | Path,
    *,
    required_archives: Sequence[str] = REQUIRED_TEACH_ARCHIVES,
) -> tuple[dict[str, Any], ...]:
    root = Path(directory)
    rows: list[dict[str, Any]] = []
    for archive in required_archives:
        path = root / archive
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        rows.append(
            {
                "archive": archive,
                "path": path.as_posix(),
                "exists": exists,
                "bytes": size,
                "sha256": sha256_file(path) if exists and size else None,
                "nonempty": bool(exists and size),
            }
        )
    return tuple(rows)


def _normalize_probe(row: Mapping[str, Any]) -> dict[str, Any]:
    url = row.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("TEACh access probes require an HTTPS URL")
    raw_status = row.get("status_code")
    if raw_status is None:
        status_code = None
    elif type(raw_status) is int and 100 <= raw_status <= 599:
        status_code = raw_status
    else:
        raise ValueError("TEACh access probe status_code must be HTTP-like or null")
    raw_length = row.get("content_length")
    if raw_length is None:
        content_length = None
    else:
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError) as error:
            raise ValueError("TEACh probe content_length must be an integer") from error
        if content_length < 0:
            raise ValueError("TEACh probe content_length must be nonnegative")
    error = row.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("TEACh access probe error must be text or null")
    return {
        "url": url,
        "status_code": status_code,
        "content_length": content_length,
        "error": error,
    }


def _archive_remote_status(rows: Sequence[Mapping[str, Any]]) -> str:
    if any(
        row["status_code"] in {200, 206}
        and (row["content_length"] is None or row["content_length"] > 0)
        for row in rows
    ):
        return "accessible"
    status_codes = [row["status_code"] for row in rows]
    if status_codes and all(value == 403 for value in status_codes):
        return "access_denied"
    if status_codes and all(value == 404 for value in status_codes):
        return "not_found"
    return "unverifiable"


def build_teach_access_report(
    *,
    observed_at_utc: str,
    official_commit: str,
    downloader_source: str,
    issue: Mapping[str, Any],
    issue_comments: Iterable[Mapping[str, Any]],
    remote_probes: Iterable[Mapping[str, Any]],
    local_archives: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a fail-closed access report without promoting it to data evidence."""

    if not isinstance(observed_at_utc, str) or not observed_at_utc.strip():
        raise ValueError("observed_at_utc must be a nonempty timestamp")
    if _COMMIT_PATTERN.fullmatch(official_commit) is None:
        raise ValueError("official TEACh commit must be a 40-character lowercase SHA")
    downloader = parse_official_teach_downloader(downloader_source)
    probes = tuple(_normalize_probe(row) for row in remote_probes)
    expected_urls = {
        archive: official_archive_urls(downloader["bucket"], archive)
        for archive in REQUIRED_TEACH_ARCHIVES
    }
    expected_flat = {url for urls in expected_urls.values() for url in urls}
    observed_urls = [row["url"] for row in probes]
    if len(observed_urls) != len(set(observed_urls)):
        raise ValueError("TEACh access probes contain duplicate URLs")
    if set(observed_urls) != expected_flat:
        raise ValueError("TEACh access probes do not exactly cover required official URLs")
    by_url = {row["url"]: row for row in probes}
    archive_remote: list[dict[str, Any]] = []
    for archive in REQUIRED_TEACH_ARCHIVES:
        rows = [by_url[url] for url in expected_urls[archive]]
        archive_remote.append(
            {
                "archive": archive,
                "status": _archive_remote_status(rows),
                "probes": rows,
            }
        )
    remote_statuses = [row["status"] for row in archive_remote]
    if all(value == "accessible" for value in remote_statuses):
        remote_status = "accessible"
    elif all(value == "access_denied" for value in remote_statuses):
        remote_status = "access_denied"
    else:
        remote_status = "partial_or_unverifiable"

    local_rows = tuple(dict(row) for row in local_archives)
    by_archive = {str(row.get("archive")): row for row in local_rows}
    if set(by_archive) != set(REQUIRED_TEACH_ARCHIVES):
        raise ValueError("local TEACh inventory must cover exactly the required archives")
    local_complete = all(bool(by_archive[name].get("nonempty")) for name in REQUIRED_TEACH_ARCHIVES)

    issue_number = issue.get("number")
    issue_url = issue.get("html_url")
    issue_state = issue.get("state")
    if issue_number != 37 or issue_url != OFFICIAL_TEACH_ACCESS_ISSUE:
        raise ValueError("TEACh access report is not bound to official issue #37")
    if issue_state not in {"open", "closed"}:
        raise ValueError("official TEACh issue has an invalid state")
    comments = tuple(dict(row) for row in issue_comments)
    maintainer_comments = [
        row
        for row in comments
        if str(row.get("author_association", "")).upper() in _MAINTAINER_ASSOCIATIONS
    ]

    if local_complete:
        decision = "local_archives_present_provenance_unverified"
        reason = (
            "Required local files exist, but the official downloader publishes no "
            "expected hashes; archive origin must be independently recorded before use."
        )
    elif remote_status == "accessible":
        decision = "official_download_endpoints_accessible"
        reason = "Both required archives have at least one successful official HEAD endpoint."
    elif remote_status == "access_denied":
        decision = "blocked_by_official_host"
        reason = "Both documented URL forms return HTTP 403 for both required archives."
    else:
        decision = "access_unverifiable"
        reason = "The required official endpoints were not uniformly accessible or denied."

    return {
        "schema_version": 1,
        "observed_at_utc": observed_at_utc.strip(),
        "claim_scope": "official data-access and provenance audit; not dataset or model evidence",
        "official_source": {
            "repository": OFFICIAL_TEACH_REPOSITORY,
            "commit": official_commit,
            "downloader_path": OFFICIAL_TEACH_DOWNLOADER_PATH,
            "downloader_sha256": sha256_bytes(downloader_source.encode("utf-8")),
            "bucket": downloader["bucket"],
            "downloader_file_list": list(downloader["file_list"]),
            "required_archives": list(REQUIRED_TEACH_ARCHIVES),
        },
        "official_issue": {
            "number": issue_number,
            "url": issue_url,
            "state": issue_state,
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "comments": len(comments),
            "maintainer_comments": len(maintainer_comments),
            "maintainer_response_present": bool(maintainer_comments),
        },
        "remote": {
            "method": "HEAD only; no archive bodies requested",
            "status": remote_status,
            "archives": archive_remote,
        },
        "local": {
            "required_archives_present": local_complete,
            "provenance_verified": False,
            "archives": [by_archive[name] for name in REQUIRED_TEACH_ARCHIVES],
        },
        "decision": {
            "status": decision,
            "reason": reason,
            "manifest_preparation_ready": False,
            "layer_a_evidence_available": False,
            "layer_b_evidence_available": False,
            "layer_c_evidence_available": False,
            "performance_evidence": False,
        },
    }


def write_teach_access_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
