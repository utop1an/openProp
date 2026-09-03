from openprop.neural_persistence import NeuralPersistenceModel
from openprop.synthetic_persistence import contextual_location_data


HOUR = 60 * 60


def main() -> None:
    examples = contextual_location_data(samples_per_context=300)
    observed = sum(example.event_observed for example in examples)
    censored = len(examples) - observed
    training = NeuralPersistenceModel.fit(
        examples,
        epochs=300,
        learning_rate=0.02,
        depth=3,
        hidden_dim=32,
    )
    model = training.model

    table_features = ("location", "cup", "on", "table", "kitchen")
    cabinet_features = ("location", "cup", "inside", "cabinet", "kitchen")
    table_hazard = model.hazard_per_hour(table_features)
    cabinet_hazard = model.hazard_per_hour(cabinet_features)
    table_survival = model.survival_probability(
        property_name="location",
        subject_type="cup",
        state_predicate="on",
        context_object="table",
        scene="kitchen",
        duration_seconds=5 * HOUR,
    )
    cabinet_survival = model.survival_probability(
        property_name="location",
        subject_type="cup",
        state_predicate="inside",
        context_object="cabinet",
        scene="kitchen",
        duration_seconds=5 * HOUR,
    )

    print(f"examples: {len(examples)} (events={observed}, censored={censored})")
    print(f"loss: {training.initial_loss:.4f} -> {training.final_loss:.4f}")
    print(f"on(table) hazard/hour:       {table_hazard:.4f}")
    print(f"inside(cabinet) hazard/hour: {cabinet_hazard:.4f}")
    print(f"on(table), after 5h:         {table_survival:.4f}")
    print(f"inside(cabinet), after 5h:   {cabinet_survival:.4f}")


if __name__ == "__main__":
    main()
