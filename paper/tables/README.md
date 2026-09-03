# Evidence-locked paper tables

Build the seven current controlled, secondary, and claim-boundary tables from the
verified paper claim manifest:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_paper_tables.py
```

The generator first runs the same hash and exact-metric verification used by
`scripts/verify_paper_claims.py`. It then rejects any table value whose JSON
pointer is not explicitly bound to the corresponding claim. Generated Markdown
supports review inside the repository and is inserted between generated-table
markers in `paper/manuscript.md`. Generated LaTeX uses `booktabs` and `tabularx`
without vertical rules; include both packages in the paper preamble.

The outputs are:

- `controlled_compositional_results.{md,tex}`: five-seed synthetic held-out-
  tuple mechanism validation with mean and standard deviation;
- `typed_component_ablation.{md,tex}`: paired NLL contribution of subject,
  relation, and scene on identical held-out-tuple test rows;
- `controlled_decision_utility.{md,tex}`: matched factor-removal Top-1 on 40
  axis-isolated cases and ten untouched confirmation seeds;
- `observation_process_bias.{md,tex}`: five-seed inspection-frequency mechanism
  validation with hazard error, schedule gap, and exact-time test likelihood;
- `observation_grounding_decisions.{md,tex}`: frozen ten-seed confirmation that
  interval semantics remove an inspection-conditioned Top-1 and target-scene
  disparity on 40 balanced analytic decisions;
- `external_language_results.{md,tex}`: train-only ALFRED retrieval, positive-
  evidence fusion, paired task-clustered intervals, and a frozen local-parser
  comparison, explicitly limited to language-to-frame parsing;
- `claim_boundaries.{md,tex}`: contradicted neural-necessity and adaptation-
  safety claims plus the pending semi-real grounding claim; and
- `table_manifest.json`: source hashes, every consumed claim-bound pointer,
  verification counts, and output hashes.

Do not edit generated files directly. These seven tables are not a substitute for
the pending official TEACh table and cannot support a semi-real or real-world
effectiveness claim.
