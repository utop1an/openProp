# Related-work and novelty audit

Status: frozen positioning audit for the 2026-08-26 working draft. This file
defines comparison boundaries; it is not experimental evidence.

## Defensible novelty statement

OpenProp does **not** propose a new visual mapper, scene-graph constructor,
general-purpose memory, active-search policy, or survival estimator. It defines
and audits a narrower interface: given a language query, a candidate set, and
timestamped heterogeneous observations, rank the candidate whose *current*
properties are best supported while preserving value types, missingness,
provenance, and censoring semantics. Its contribution is the combination of:

1. a typed and decomposable current-evidence score;
2. a hard boundary between semantic parsing and deterministic scoring;
3. context-dependent persistence learned from interval- and right-censored
   histories; and
4. an evaluation protocol in which hidden current truth never enters matching.

The strongest adjacent systems are therefore complementary upstream memories
or broader embodied agents, not interchangeable implementations of this scoring
contract.

## Competitor matrix

| Work | Primary capability | Treatment of change or history | Closest overlap | Exact OpenProp distinction | Experimental implication |
|---|---|---|---|---|---|
| [OVSG / context-aware entity grounding](https://proceedings.mlr.press/v229/chang23b.html) | Grounds free-form queries in an open-vocabulary 3D scene graph using contextual relations. | Queries a constructed scene representation. | Language-conditioned entity selection and relation-aware context. | OpenProp assumes candidates and observations are supplied, then exposes typed match, missing coverage, confidence, and evidence validity rather than constructing the 3D graph. | Cite as the closest static/contextual grounding boundary; do not claim superior localization without a shared visual benchmark. |
| [ConceptGraphs](https://arxiv.org/abs/2309.16650) and [Open3DSG](https://openaccess.thecvf.com/content/CVPR2024/html/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.html) | Build queryable open-vocabulary 3D object and relation representations. | Primarily representation construction from visual observations. | Supply candidate entities and semantic relations that OpenProp could consume. | OpenProp is an evidence-ranking layer, not a mapping or representation contribution. | Treat their outputs as a future upstream integration, not as executable scoring baselines on the current artifacts. |
| [Scene Graph Memory](https://proceedings.mlr.press/v202/kurenkov23a.html) | Accumulates observations in a partially observed dynamic graph and predicts object locations for search. | Models dynamic object-location uncertainty in an accumulated graph. | Historical memory under partial observability. | OpenProp asks which typed historical facts still support a current candidate and provides a per-property score audit; it does not learn a search policy or location predictor. | Compare task definitions and evidence interfaces, not raw success rates from different simulators. |
| [DynaMem](https://arxiv.org/abs/2411.04999) | Incrementally updates a spatio-semantic 3D memory as objects appear, disappear, or move, and uses it for localization and manipulation. | Revises a dynamic 3D representation online. | Current object retrieval in changing environments. | OpenProp does not replace dynamic map updates; it explicitly scores the residual validity of timestamped typed evidence, including unobserved intervals and missing properties. | A fair future integration should feed DynaMem-like observations into OpenProp and compare ranking ablations under identical candidates. |
| [3D-Mem](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_3D-Mem_3D_Scene_Memory_for_Embodied_Exploration_and_Reasoning_CVPR_2025_paper.html), [Embodied VideoAgent](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Embodied_VideoAgent_Persistent_Memory_from_Egocentric_Videos_and_Embodied_Sensors_ICCV_2025_paper.html), and [GraphEQA](https://proceedings.mlr.press/v305/saxena25a.html) | Maintain multimodal scene memories for exploration, question answering, and reasoning. | Preserve or update observations across views and time for downstream VLM reasoning. | Persistent embodied memory and task-conditioned recall. | OpenProp isolates a deterministic, typed decision after memory retrieval; it does not claim a complete multimodal reasoning architecture. | Do not compare end-task accuracy until the same perception, candidates, and action interface are controlled. |
| [Mimir](https://arxiv.org/abs/2608.04933) (concurrent) | Separates world and task memory, then dynamically binds recalled world candidates and evidence to the active goal before planning. | Maintains object state, location, evidence, execution progress, and constraints across long-horizon tasks. | The phrase *dynamic grounding*, explicit evidence, and candidate binding create the strongest naming-level overlap. | Mimir is an end-to-end neuro-symbolic memory-and-execution system; OpenProp's narrower claim is a typed, censoring-aware and auditable current-evidence score over supplied histories. | Avoid priority claims over “dynamic grounding.” Compare factorization, interfaces, and evaluation leakage; wait for code before proposing an executable baseline. |
| [STAR](https://arxiv.org/abs/2511.14004) and [DGSG-Mind](https://arxiv.org/abs/2605.29879) (concurrent) | Couple memory with temporal/spatial search, or dynamic Gaussian scene graphs with multimodal reasoning. | Actively search over time or update object-level 3D topology. | Open-world retrieval under environmental change. | OpenProp does not select exploration actions or reconstruct changing geometry; it corrects and audits evidence scoring after observations exist. | Discuss as concurrent scope boundaries; no apples-to-oranges leaderboard comparison. |
| [DeepHit](https://ojs.aaai.org/index.php/AAAI/article/view/11842), [Survival-CRPS](https://proceedings.mlr.press/v115/avati20a.html), and dependent-censoring models such as [Deep Copula Survival](https://ojs.aaai.org/index.php/AAAI/article/view/30047) | Estimate event-time distributions under censoring, including competing risks, interval censoring, calibration, or dependent censoring. | Treat censoring as part of event-time estimation rather than as a negative label. | Persistence estimation from incomplete observation histories. | OpenProp contributes neither a general survival objective nor censoring theory; it installs censoring-correct persistence behind a typed grounding interface and measures the downstream ranking consequences of inspection bias. | Include standard fixed, exponential, Weibull, Cox where legal, and observation-aware estimators; avoid survival-method novelty claims. |

## Positioning rules for the paper

- Say **complementary to dynamic memory**, not “the first dynamic memory” or
  “prior systems ignore time.”
- Say **current-evidence ranking over supplied candidates**, not generic visual
  grounding, object navigation, or end-to-end embodied grounding.
- Use *dynamic grounding* only with a qualifier such as *typed current-evidence
  scoring* because Mimir independently uses the broader term.
- Present synthetic experiments as mechanism validation. Only the frozen TEACh
  protocol can support a semi-real longitudinal effectiveness claim.
- Do not place upstream mapping, search, question-answering, or manipulation
  success rates in the same quantitative table as OpenProp scores.
- The executable baseline family for the current task is no decay, fixed decay,
  global survival, typed factorized survival, and observation-aware variants
  under identical queries, observations, candidates, and evaluation truth.

## Remaining citation and baseline risk

The literature is moving quickly and several closest works are concurrent
preprints. Before submission, repeat the title/abstract audit, check whether
Mimir and DGSG-Mind have released code or peer-reviewed versions, and update the
comparison matrix without changing the frozen experimental task. The official
TEACh result must compare scoring variants on identical upstream observations;
it must not imply an end-to-end comparison with systems whose perception and
action stacks differ.
