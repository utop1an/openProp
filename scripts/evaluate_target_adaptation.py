from __future__ import annotations

import argparse
import hashlib
import math
import random
import json
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.latent_mechanism_shift import (
    MECHANISM_CONDITIONS,
    latent_mechanism_shift_data,
)
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from openprop.synthetic_survival_oracle import SyntheticWeibullOracle
from openprop.target_adaptation import (
    build_target_calibration_protocol,
    fit_log_risk_affine_adapter,
    select_sign_gated_model,
)


METHODS = (
    "source",
    "scale_only",
    "affine_log_risk",
    "sign_gated",
    "target_per_context",
)
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }

def _paired_inference(
    values: list[float],
    *,
    key: str,
    bootstrap_resamples: int = 20_000,
) -> dict[str, float | int]:
    """Seed-cluster bootstrap CI and two-sided exact sign test."""

    if not values or bootstrap_resamples <= 0:
        raise ValueError("paired inference requires values and resamples")
    seed = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    rng = random.Random(seed)
    sample_count = len(values)
    bootstrap_means = sorted(
        statistics.fmean(rng.choices(values, k=sample_count))
        for _ in range(bootstrap_resamples)
    )
    lower_index = int(0.025 * (bootstrap_resamples - 1))
    upper_index = int(0.975 * (bootstrap_resamples - 1))
    wins = sum(value > 1e-12 for value in values)
    losses = sum(value < -1e-12 for value in values)
    ties = sample_count - wins - losses
    non_ties = wins + losses
    if non_ties:
        tail = min(wins, losses)
        sign_p = min(
            1.0,
            2.0
            * sum(math.comb(non_ties, index) for index in range(tail + 1))
            / (2.0**non_ties),
        )
    else:
        sign_p = 1.0
    return {
        "bootstrap_95_ci_lower": bootstrap_means[lower_index],
        "bootstrap_95_ci_upper": bootstrap_means[upper_index],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "two_sided_exact_sign_p": sign_p,
    }



def _metrics(report) -> dict[str, float]:
    return {
        "negative_log_likelihood": report.negative_log_likelihood,
        "concordance_index": report.concordance_index,
        "integrated_brier_score": report.integrated_brier_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate leakage-safe target calibration under latent mechanism shift."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[31, 41, 53, 67, 79, 97, 109, 127, 149, 173],
    )
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument(
        "--calibration-sizes-per-context",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--evaluation-horizons",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 8.0, 12.0],
    )
    parser.add_argument("--source-epochs", type=int, default=1200)
    parser.add_argument("--adapter-epochs", type=int, default=1000)
    parser.add_argument("--adapter-learning-rate", type=float, default=0.03)
    parser.add_argument("--adapter-slope-l2", type=float, default=1e-4)
    parser.add_argument("--target-prior-exposure-hours", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/target_adaptation_results.json"),
    )
    args = parser.parse_args()
    sizes = args.calibration_sizes_per_context
    horizons = tuple(args.evaluation_horizons)
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or sizes != sorted(set(sizes))
        or not sizes
        or sizes[0] <= 0
        or args.samples_per_context <= sizes[-1]
        or args.source_epochs <= 0
        or args.adapter_epochs <= 0
        or args.adapter_learning_rate <= 0
        or args.adapter_slope_l2 < 0
        or args.target_prior_exposure_hours < 0
        or list(horizons) != sorted(set(horizons))
        or any(value <= 0 or value >= 16.0 for value in horizons)
    ):
        parser.error("invalid seeds, sample sizes, horizons, or optimizer settings")

    runs: list[dict[str, object]] = []
    max_calibration = sizes[-1]
    for seed in args.seeds:
        dataset = latent_mechanism_shift_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train,
            epochs=args.source_epochs,
        )
        source.calibrate(dataset.validation)
        protocol = build_target_calibration_protocol(
            dataset,
            max_calibration_per_context=max_calibration,
            split_seed=seed + 5_000_003,
        )
        for condition in MECHANISM_CONDITIONS:
            test_rows = protocol.tests[condition]
            oracle = SyntheticWeibullOracle(
                dataset.test_hazards[condition],
                float(MECHANISM_CONDITIONS[condition]["weibull_shape"]),
            )
            oracle_metrics = _metrics(
                evaluate_survival_advanced(
                    oracle,
                    test_rows,
                    horizons_hours=horizons,
                )
            )
            source_metrics = _metrics(
                evaluate_survival_advanced(
                    source,
                    test_rows,
                    horizons_hours=horizons,
                )
            )
            for samples_per_context in sizes:
                calibration = protocol.calibration_subset(
                    condition,
                    samples_per_context,
                )
                scale = fit_log_risk_affine_adapter(
                    source,
                    calibration,
                    fit_slope=False,
                    epochs=args.adapter_epochs,
                    learning_rate=args.adapter_learning_rate,
                    slope_l2_penalty=args.adapter_slope_l2,
                )
                affine = fit_log_risk_affine_adapter(
                    source,
                    calibration,
                    fit_slope=True,
                    epochs=args.adapter_epochs,
                    learning_rate=args.adapter_learning_rate,
                    slope_l2_penalty=args.adapter_slope_l2,
                )
                sign_gated = select_sign_gated_model(source, affine)
                target_only = PerContextExponentialPersistenceModel.fit(
                    calibration,
                    prior_exposure_hours=args.target_prior_exposure_hours,
                )
                models = {
                    "source": source,
                    "scale_only": scale,
                    "affine_log_risk": affine,
                    "sign_gated": sign_gated,
                    "target_per_context": target_only,
                }
                model_metrics = {
                    name: _metrics(
                        evaluate_survival_advanced(
                            model,
                            test_rows,
                            horizons_hours=horizons,
                        )
                    )
                    for name, model in models.items()
                }
                if model_metrics["source"] != source_metrics:
                    raise RuntimeError("source test metrics changed across sample sizes")
                runs.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "calibration_samples_per_context": samples_per_context,
                        "calibration_examples": len(calibration),
                        "test_examples": len(test_rows),
                        "models": model_metrics,
                        "oracle": oracle_metrics,
                        "adapter": {
                            "scale_only": {
                                "slope": scale.slope,
                                "initial_calibration_nll": (
                                    scale.initial_negative_log_likelihood
                                ),
                                "final_calibration_nll": (
                                    scale.final_negative_log_likelihood
                                ),
                            },
                            "affine_log_risk": {
                                "slope": affine.slope,
                                "initial_calibration_nll": (
                                    affine.initial_negative_log_likelihood
                                ),
                                "final_calibration_nll": (
                                    affine.final_negative_log_likelihood
                                ),
                            },
                            "sign_gated_selected": (
                                "affine_log_risk" if sign_gated is affine
                                else "source"
                            ),
                        },
                    }
                )
                print(
                    f"seed={seed} condition={condition:26s} "
                    f"k={samples_per_context:2d} slope={affine.slope:+.3f} "
                    f"source-C={source_metrics['concordance_index']:.3f} "
                    f"affine-C={model_metrics['affine_log_risk']['concordance_index']:.3f}"
                )

    def metric_values(
        condition: str,
        size: int,
        method: str,
        metric: str,
    ) -> list[float]:
        return [
            float(run["models"][method][metric])  # type: ignore[index]
            for run in runs
            if run["condition"] == condition
            and run["calibration_samples_per_context"] == size
        ]

    aggregate = {
        condition: {
            str(size): {
                method: {
                    metric: _summary(metric_values(condition, size, method, metric))
                    for metric in METRICS
                }
                for method in METHODS
            }
            for size in sizes
        }
        for condition in MECHANISM_CONDITIONS
    }
    paired_delta_vs_source: dict[str, object] = {}
    for condition in MECHANISM_CONDITIONS:
        paired_delta_vs_source[condition] = {}
        for size in sizes:
            paired_delta_vs_source[condition][str(size)] = {}
            for method in METHODS[1:]:
                paired_delta_vs_source[condition][str(size)][method] = {}
                for metric in METRICS:
                    source_values = metric_values(
                        condition, size, "source", metric
                    )
                    adapted_values = metric_values(
                        condition, size, method, metric
                    )
                    deltas = [
                        (adapted - source)
                        if metric == "concordance_index"
                        else (source - adapted)
                        for source, adapted in zip(
                            source_values, adapted_values, strict=True
                        )
                    ]
                    paired_delta_vs_source[condition][str(size)][method][metric] = {
                        **_summary(deltas),
                        **_paired_inference(
                            deltas,
                            key=f"{condition}|{size}|{method}|{metric}",
                        ),
                        "direction": "positive means adapted is better",
                    }
    affine_slopes = {
        condition: {
            str(size): {
                **_summary(
                    [
                        float(run["adapter"]["affine_log_risk"]["slope"])  # type: ignore[index]
                        for run in runs
                        if run["condition"] == condition
                        and run["calibration_samples_per_context"] == size
                    ]
                ),
                "negative_count": sum(
                    float(run["adapter"]["affine_log_risk"]["slope"]) < 0  # type: ignore[index]
                    for run in runs
                    if run["condition"] == condition
                    and run["calibration_samples_per_context"] == size
                ),
            }
            for size in sizes
        }
        for condition in MECHANISM_CONDITIONS
    }
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "source_samples_per_context": args.samples_per_context,
            "calibration_sizes_per_context": sizes,
            "target_context_count": len(dataset.test_hazards["in_distribution"]),
            "fixed_test_examples_per_condition": len(
                protocol.tests["in_distribution"]
            ),
            "conditions": MECHANISM_CONDITIONS,
            "split": (
                "SHA-256 rank of group_id and split seed only; maximum pool frozen "
                "before outcomes; nested calibration subsets; common held-out test"
            ),
            "source_fit": "source train; global scale on source validation",
            "target_fit": (
                "each adapter fits only its labeled target calibration subset; "
                "no hyperparameter or method selection on target test"
            ),
            "affine_adapter": (
                "log h_target = intercept + slope * centered log source risk; "
                "fixed optimizer and slope L2 for every condition and sample size"
            ),
            "inference": (
                "paired deltas clustered by seed; deterministic 20000-resample "
                "percentile bootstrap 95% CI; two-sided exact sign test excludes "
                "ties"
            ),
            "sign_gate": (
                "select affine repair iff its target-calibration slope is negative; "
                "otherwise preserve the source model; never consult target test"
            ),
            "target_per_context": (
                "strong transductive target-only MLE that sees the same context "
                "identities in calibration and test"
            ),
            "evaluation_horizons_hours": args.evaluation_horizons,
            "oracle": "generator truth is evaluation-only",
            "claim_scope": "synthetic target-adaptation mechanism validation",
        },
        "aggregate": aggregate,
        "paired_delta_vs_source": paired_delta_vs_source,
        "affine_slopes": affine_slopes,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
