# ICLR 2027 submission build

This directory contains the anonymous OpenProp submission source and an integrity-pinned copy of the official ICLR 2027 style files.

## Submission gates

- The submission remains anonymous; `\\iclrfinalcopy` is disabled.
- Main text ends at `\\label{sec:main-end}` and must end on page 9 or earlier.
- AI-use, ethics, and reproducibility statements follow the main text.
- References and appendices are outside the main-text page budget.
- Unresolved references, citations, placeholders, and template drift fail the build.
- Synthetic results are described only as mechanism validation.

## Build

```powershell
python scripts/build_iclr2027_submission.py --tectonic tmp/pdfs/tectonic-0.17.0/tectonic.exe
```

The validated PDF is written to `output/pdf/OpenProp_ICLR2027_submission.pdf`.

Official author guidelines: <https://iclr.cc/Conferences/2027/AuthorGuidelines>
