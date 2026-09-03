from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openprop.teach_access import (
    OFFICIAL_TEACH_ACCESS_ISSUE,
    OFFICIAL_TEACH_COMMIT,
    OFFICIAL_TEACH_DOWNLOADER_PATH,
    REQUIRED_TEACH_ARCHIVES,
    build_teach_access_report,
    inspect_local_teach_archives,
    official_archive_urls,
    parse_official_teach_downloader,
    write_teach_access_report,
)


USER_AGENT = "OpenProp-TEACh-access-audit/1"


def _request(url: str, *, method: str, timeout: float) -> tuple[bytes, Any, int]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers, int(response.status)


def _json_url(url: str, *, timeout: float) -> Any:
    body, _, _ = _request(url, method="GET", timeout=timeout)
    return json.loads(body.decode("utf-8"))


def _head(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        _, headers, status = _request(url, method="HEAD", timeout=timeout)
        raw_length = headers.get("Content-Length")
        return {
            "url": url,
            "status_code": status,
            "content_length": int(raw_length) if raw_length is not None else None,
            "error": None,
        }
    except urllib.error.HTTPError as error:
        raw_length = error.headers.get("Content-Length") if error.headers else None
        return {
            "url": url,
            "status_code": int(error.code),
            "content_length": int(raw_length) if raw_length is not None else None,
            "error": f"HTTPError: {error.reason}",
        }
    except (OSError, urllib.error.URLError) as error:
        return {
            "url": url,
            "status_code": None,
            "content_length": None,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current official TEACh archive access without downloading archive bodies."
        )
    )
    parser.add_argument(
        "--archive-directory",
        type=Path,
        default=Path("data/teach/archives"),
        help="optional local directory inspected for the two required archives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/teach_access_audit.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--require-accessible",
        action="store_true",
        help="exit with status 2 unless official endpoints are currently accessible",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    downloader_url = (
        "https://raw.githubusercontent.com/alexa/teach/"
        f"{OFFICIAL_TEACH_COMMIT}/{OFFICIAL_TEACH_DOWNLOADER_PATH}"
    )
    downloader_body, _, _ = _request(
        downloader_url, method="GET", timeout=args.timeout_seconds
    )
    downloader_source = downloader_body.decode("utf-8")
    downloader = parse_official_teach_downloader(downloader_source)
    issue_api = "https://api.github.com/repos/alexa/teach/issues/37"
    issue = _json_url(issue_api, timeout=args.timeout_seconds)
    comments = _json_url(issue_api + "/comments?per_page=100", timeout=args.timeout_seconds)
    probes = [
        _head(url, timeout=args.timeout_seconds)
        for archive in REQUIRED_TEACH_ARCHIVES
        for url in official_archive_urls(downloader["bucket"], archive)
    ]
    report = build_teach_access_report(
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        official_commit=OFFICIAL_TEACH_COMMIT,
        downloader_source=downloader_source,
        issue=issue,
        issue_comments=comments,
        remote_probes=probes,
        local_archives=inspect_local_teach_archives(args.archive_directory),
    )
    write_teach_access_report(args.output, report)
    print(
        json.dumps(
            {
                "decision": report["decision"]["status"],
                "remote": report["remote"]["status"],
                "local_archives_present": report["local"]["required_archives_present"],
                "issue": OFFICIAL_TEACH_ACCESS_ISSUE,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_accessible and report["remote"]["status"] != "accessible":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
