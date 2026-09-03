from __future__ import annotations

from pathlib import Path

from openprop.neural_persistence import NeuralPersistenceModel
from openprop.observation_history import (
    ObservationHistoryRecord,
    grouped_split,
    history_to_examples,
    read_history_jsonl,
    write_history_jsonl,
)
from openprop.survival_evaluation import evaluate_survival
from openprop.synthetic_persistence import contextual_location_data


ARTIFACTS = Path("artifacts")


def _history_records():
    records = []
    for index, example in enumerate(contextual_location_data(samples_per_context=300)):
        records.append(
            ObservationHistoryRecord(
                record_id=f"synthetic-{index:04d}",
                entity_id=example.group_id,
                property_name=example.property_name,
                subject_type=example.subject_type,
                state_predicate=example.state_predicate,
                context_object=example.context_object,
                scene=example.scene,
                observed_at=0.0,
                followup_at=example.duration_seconds,
                state_changed=example.event_observed,
                source="synthetic-contextual-location-v1",
            )
        )
    return tuple(records)


def _print_evaluation(label, evaluation):
    print(f"{label} NLL: {evaluation.negative_log_likelihood:.4f}")
    for horizon in evaluation.horizons:
        print(
            f"  {horizon.hours:>4g}h: n={horizon.evaluable_examples:>3d} "
            f"Brier={horizon.brier_score:.4f} ECE={horizon.expected_calibration_error:.4f}"
        )


def main() -> None:
    history_path = ARTIFACTS / "observation_history.jsonl"
    model_path = ARTIFACTS / "contextual_persistence.pt"
    write_history_jsonl(history_path, _history_records())
    records = read_history_jsonl(history_path)
    split = grouped_split(history_to_examples(records))

    training = NeuralPersistenceModel.fit(
        split.train,
        epochs=250,
        learning_rate=0.02,
        depth=3,
        hidden_dim=32,
    )
    validation_before = evaluate_survival(training.model, split.validation)
    scale = training.model.calibrate(split.validation)
    validation_after = evaluate_survival(training.model, split.validation)
    test_evaluation = evaluate_survival(training.model, split.test)
    training.model.save(model_path)
    restored = NeuralPersistenceModel.load(model_path)

    probe = ("location", "cup", "inside", "cabinet", "kitchen")
    if abs(restored.hazard_per_hour(probe) - training.model.hazard_per_hour(probe)) > 1e-7:
        raise RuntimeError("restored model prediction differs from saved model")

    print(f"records: {len(records)}")
    print(
        f"split: train={len(split.train)} validation={len(split.validation)} "
        f"test={len(split.test)}"
    )
    print(f"training loss: {training.initial_loss:.4f} -> {training.final_loss:.4f}")
    print(f"validation hazard scale: {scale:.4f}")
    _print_evaluation("validation before calibration", validation_before)
    _print_evaluation("validation after calibration", validation_after)
    _print_evaluation("test", test_evaluation)
    print(f"history: {history_path}")
    print(f"model:   {model_path}")


if __name__ == "__main__":
    main()
