# Paper figures

Both paper figures are generated from code rather than edited by hand:

```powershell
python -m pip install -e ".[paper]"
python scripts/build_paper_teaser.py
python scripts/build_paper_pipeline_figure.py
```

The commands write vector-first SVGs and PNG previews:

- `openprop_teaser.{svg,png}` for the first-page problem/result summary; and
- `openprop_task_pipeline.{svg,png}` for the detailed method and leakage
  boundaries.

The teaser carries one message: candidates can have equal typed language match
while differing in how strongly their historical observations support a current
decision. It shows the inspection-frequency failure, the interval-aware rank
repair, and only the claim-bound synthetic confirmation numbers. Candidate cards
contain no target identity; `current_truth = mug A` appears only in the dashed
evaluation box. The footer explicitly denies real-world effectiveness and keeps
perception and mapping upstream.

The pipeline figure separately exposes three architectural boundaries:

1. a dynamic memory or scene graph supplies observations, while perception and
   mapping remain upstream;
2. OpenProp maps language to a typed frame, then performs deterministic typed,
   freshness-aware scoring with explicit missingness and coverage; and
3. `current_truth` enters evaluation metrics only and never enters the matcher.

Pipeline ranking numbers are illustrative and are not benchmark results. Teaser
numbers are copied from the evidence-locked observation-grounding confirmation
and bounded as synthetic controlled evidence in both the image and caption.

After changing either generator, regenerate its SVG and PNG, visually inspect
the full-resolution PNG and a paper-width preview, and run the corresponding
paper-figure test. Never edit generated assets directly.
