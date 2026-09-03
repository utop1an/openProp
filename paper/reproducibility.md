# Computational reproducibility

OpenProp uses two complementary integrity layers:

1. `paper/claims.json` binds paper claims to exact JSON pointers and SHA-256
   hashes in result artifacts.
2. `paper/reproducibility_manifest.json` binds the computational snapshot,
   runtime versions, experiment entry points, result outputs, external audits,
   and release gates. Schema v2 keeps network/provenance audits separate from
   model experiments so infrastructure checks cannot inflate empirical counts.

The manifest is deliberately fail-closed. A changed source, paper file, result
artifact, command entry point, claim count, or recorded runtime causes
verification to fail. External ALFRED and TEACh archives are not redistributed;
their frozen preparation protocols and availability status remain explicit.

## Reproduce the checked-in controlled evidence

From the repository root in PowerShell:

```powershell
python -m pip install -e ".[ml,paper]"
$env:PYTHONPATH = "src"
python scripts/verify_reproducibility_manifest.py --require-runtime-match
python scripts/verify_paper_claims.py
python -m unittest discover -s tests -v
```

The manifest records ten experiments: eight controlled mechanism/decision
entries, the external ALFRED pipeline, and the retained negative adaptation
result. An eleventh experiment is not implied: the TEACh access snapshot is stored under
`external_audits` and carries an explicit non-performance scope. Rebuild the
seven current tables without editing them:

```powershell
python scripts/build_paper_tables.py --check
```

## Update the snapshot after an intentional change

Only rebuild after the changed code, documentation, paper, and result artifacts
have passed their focused checks:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_reproducibility_manifest.py
python scripts/verify_reproducibility_manifest.py --require-runtime-match
```

The source-tree digest excludes only the reproducibility manifest itself,
Python bytecode, and local test caches. It includes project metadata, source,
scripts, tests, examples, documentation, paper files, and every artifact used by
the claim manifest. This makes documentation and protocol drift visible rather
than treating code alone as the experiment.

## Release gate

The current desktop workspace cannot expose Git metadata, so the checked-in
manifest truthfully records `revision.status = unavailable` and
`clean_git_revision_bound = false`. The content-addressed source snapshot is
still verifiable, but this is not a release-grade revision binding. Before an
archive or submission release, run from a clean Git checkout:

```powershell
python scripts/build_reproducibility_manifest.py --require-git-revision
python scripts/verify_reproducibility_manifest.py --require-runtime-match --require-git-revision
```

Submission readiness additionally requires the official TEACh result. The
manifest therefore keeps `submission_release_ready = false`; generating a valid
computational snapshot must never silently clear the empirical submission gate.
