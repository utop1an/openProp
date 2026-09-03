# ALFRED action-history grounding feasibility audit

## Question

Can the locally available official ALFRED trajectories provide the missing
external longitudinal grounding result while TEACh archives remain
inaccessible? This audit tests that premise before a favorable but invalid
proxy result can enter the paper.

## Protocol

The deterministic audit scans all 251 valid-seen and 255 valid-unseen
`traj_data.json` files. For every object interaction it keeps only histories
strictly preceding that action, groups candidates by typed object class, and
asks whether the eventual target had previously appeared alongside at least
one distinct same-type entity. The future action target is evaluation-only.
Annotation-zero high-level human descriptions supply the prospective language
query.

AI2-THOR changes an object's raw ID when a derived child such as
`AppleSliced_1` is created. The audit therefore canonicalizes identity to the
first four `objectId` fields (type and original coordinates). Without this
lineage correction, 244 apparent multi-candidate cases are reported; most are
one entity before and after slicing. The corrected count is 70.

Run:

```powershell
python scripts/audit_alfred_longitudinal_feasibility.py `
  --root artifacts/external/alfred/json_2.1.0 `
  --splits valid_seen valid_unseen `
  --output artifacts/alfred_longitudinal_feasibility.json
```

## Verified counts

| Quantity | Count |
|---|---:|
| Validation trajectories | 506 |
| Object-interaction steps | 3,327 |
| Initially same-type-ambiguous steps | 1,245 |
| Repeated-target steps after lineage correction | 2,132 |
| Distinct same-type multi-history cases | 70 |
| Valid-seen / valid-unseen eligible cases | 34 / 36 |
| `pick_two_obj_and_place` eligible cases | 64 |
| Unique most-recent target / recency wrong | 68 / 2 |

## Decision

The 70 corrected cases do not replace TEACh. Sixty-four come from
`pick_two_obj_and_place`, and 68 targets are the object just picked up before it
is put down. A 97.1% most-recent baseline would therefore measure short-range
action autocorrelation or held-object state, not persistence of stale perceptual
evidence. The slice is useful for future action-history coreference work, but
promoting it to the central external result would misstate the task.

The official TEACh path remains the required longitudinal benchmark. This
negative audit narrows that requirement and prevents a high-looking proxy score
from weakening the paper's scientific validity.
