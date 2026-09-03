# OpenProp paper package

This directory is the evidence-locked working package for a competitive ICLR
submission. It is deliberately stricter than the project documentation: a
result claim is paper-eligible only when `claims.json` names its scope, status,
artifact hash, and exact metric checks.

## Current status

- The thesis, claim hierarchy, and forbidden claims are frozen in
  [claim-hierarchy.md](claim-hierarchy.md).
- The working paper story and section-level experiment plan are in
  [manuscript.md](manuscript.md).
- The claim-to-population, baseline, analysis-unit, and scope contract behind
  Section 4 is frozen in [experimental-protocol-audit.md](experimental-protocol-audit.md).
- The Discussion, Limitations/Broader Impact, and Conclusion are reverse-outlined
  and claim-audited in [closing-section-audit.md](closing-section-audit.md).
- The reproducible vector-first teaser and task/method overview are in
  [figures/](figures/README.md).
- Evidence-locked Markdown and LaTeX result/boundary tables are generated in
  [tables/](tables/README.md).
- The source snapshot, runtime, experiment commands, external inputs, and release
  gates are bound by the [computational reproducibility package](reproducibility.md)
  and `reproducibility_manifest.json`.
- The strongest current rejection case and required repairs are in
  [adversarial-review.md](adversarial-review.md).
- The official TEACh longitudinal result remains the submission-critical
  external-validity gate. Until it exists, this package is a research draft,
  not a submission-ready paper.

## Verify every result claim

```powershell
$env:PYTHONPATH = "src"
python scripts/verify_paper_claims.py
```

The command fails closed if an evidence file changes, a JSON metric moves, a
path escapes the repository, or a supported claim lacks machine-checkable
evidence.


Verify the broader computational snapshot and regenerate every table in a
temporary directory:

```powershell
python scripts/verify_reproducibility_manifest.py --require-runtime-match
python scripts/build_paper_tables.py --check
python -m unittest discover -s tests -v
```

The current snapshot is content-addressed but cannot bind a Git commit because
the desktop workspace does not expose repository metadata. A submission archive
must rebuild and verify it from a clean checkout with `--require-git-revision`;
the official TEACh result remains a separate empirical release gate.
