import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_access import (
    OFFICIAL_TEACH_ACCESS_ISSUE,
    OFFICIAL_TEACH_COMMIT,
    REQUIRED_TEACH_ARCHIVES,
    build_teach_access_report,
    inspect_local_teach_archives,
    official_archive_urls,
    parse_official_teach_downloader,
)


DOWNLOADER = '''
DEFAULT_DATASET_BUCKET_NAME = "teach-dataset"
FILE_LIST = [
    "all_games.tar.gz",
    "edh_instances.tar.gz",
    "images_and_states.tar.gz",
]
'''
COMMIT = "9" * 40


class TeachAccessTests(unittest.TestCase):
    def _issue(self, *, state: str = "open") -> dict:
        return {
            "number": 37,
            "html_url": OFFICIAL_TEACH_ACCESS_ISSUE,
            "state": state,
            "created_at": "2026-07-13T05:39:23Z",
            "updated_at": "2026-08-05T08:50:38Z",
        }

    def _probes(self, status_code: int | None, *, content_length: int | None = None):
        return [
            {
                "url": url,
                "status_code": status_code,
                "content_length": content_length,
                "error": None if status_code else "network unavailable",
            }
            for archive in REQUIRED_TEACH_ARCHIVES
            for url in official_archive_urls("teach-dataset", archive)
        ]

    def _local(self, directory: Path):
        return inspect_local_teach_archives(directory)

    def test_parses_official_bucket_and_required_archive_inventory(self) -> None:
        parsed = parse_official_teach_downloader(DOWNLOADER)
        self.assertEqual(parsed["bucket"], "teach-dataset")
        self.assertEqual(
            set(REQUIRED_TEACH_ARCHIVES).issubset(parsed["file_list"]), True
        )
        with self.assertRaisesRegex(ValueError, "omits required archives"):
            parse_official_teach_downloader(
                'DEFAULT_DATASET_BUCKET_NAME="x"\nFILE_LIST=["all_games.tar.gz"]\n'
            )

    def test_all_403_is_a_blocker_but_never_performance_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_teach_access_report(
                observed_at_utc="2026-08-27T00:00:00+00:00",
                official_commit=COMMIT,
                downloader_source=DOWNLOADER,
                issue=self._issue(),
                issue_comments=[{"author_association": "NONE"}],
                remote_probes=self._probes(403),
                local_archives=self._local(Path(directory)),
            )
        self.assertEqual(report["remote"]["status"], "access_denied")
        self.assertEqual(report["decision"]["status"], "blocked_by_official_host")
        self.assertFalse(report["decision"]["manifest_preparation_ready"])
        self.assertFalse(report["decision"]["performance_evidence"])
        self.assertFalse(report["official_issue"]["maintainer_response_present"])

    def test_successful_heads_are_accessible_but_not_dataset_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_teach_access_report(
                observed_at_utc="2026-08-27T00:00:00+00:00",
                official_commit=COMMIT,
                downloader_source=DOWNLOADER,
                issue=self._issue(state="closed"),
                issue_comments=[{"author_association": "MEMBER"}],
                remote_probes=self._probes(200, content_length=100),
                local_archives=self._local(Path(directory)),
            )
        self.assertEqual(report["remote"]["status"], "accessible")
        self.assertEqual(
            report["decision"]["status"], "official_download_endpoints_accessible"
        )
        self.assertFalse(report["decision"]["layer_a_evidence_available"])
        self.assertTrue(report["official_issue"]["maintainer_response_present"])

    def test_network_failures_and_unverified_local_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unavailable = build_teach_access_report(
                observed_at_utc="2026-08-27T00:00:00+00:00",
                official_commit=COMMIT,
                downloader_source=DOWNLOADER,
                issue=self._issue(),
                issue_comments=[],
                remote_probes=self._probes(None),
                local_archives=self._local(root),
            )
            self.assertEqual(unavailable["decision"]["status"], "access_unverifiable")
            for archive in REQUIRED_TEACH_ARCHIVES:
                (root / archive).write_bytes((archive + "\n").encode("utf-8"))
            local = build_teach_access_report(
                observed_at_utc="2026-08-27T00:00:00+00:00",
                official_commit=COMMIT,
                downloader_source=DOWNLOADER,
                issue=self._issue(),
                issue_comments=[],
                remote_probes=self._probes(None),
                local_archives=self._local(root),
            )
        self.assertEqual(
            local["decision"]["status"],
            "local_archives_present_provenance_unverified",
        )
        self.assertTrue(local["local"]["required_archives_present"])
        self.assertFalse(local["local"]["provenance_verified"])
        self.assertFalse(local["decision"]["manifest_preparation_ready"])

    def test_probe_population_must_exactly_match_official_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probes = self._probes(403)
            probes.pop()
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                build_teach_access_report(
                    observed_at_utc="2026-08-27T00:00:00+00:00",
                    official_commit=COMMIT,
                    downloader_source=DOWNLOADER,
                    issue=self._issue(),
                    issue_comments=[],
                    remote_probes=probes,
                    local_archives=self._local(Path(directory)),
                )

    def test_checked_in_snapshot_is_bound_to_current_official_incident(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "artifacts" / "teach_access_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["official_source"]["commit"], OFFICIAL_TEACH_COMMIT)
        self.assertEqual(report["official_issue"]["number"], 37)
        self.assertEqual(report["official_issue"]["state"], "open")
        self.assertFalse(report["official_issue"]["maintainer_response_present"])
        self.assertEqual(report["remote"]["status"], "access_denied")
        probes = [
            probe
            for archive in report["remote"]["archives"]
            for probe in archive["probes"]
        ]
        self.assertEqual(len(probes), 4)
        self.assertEqual({probe["status_code"] for probe in probes}, {403})
        self.assertFalse(report["local"]["required_archives_present"])
        self.assertEqual(report["decision"]["status"], "blocked_by_official_host")
        for key in ("layer_a_evidence_available", "layer_b_evidence_available", "layer_c_evidence_available", "performance_evidence"):
            self.assertFalse(report["decision"][key])


if __name__ == "__main__":
    unittest.main()
