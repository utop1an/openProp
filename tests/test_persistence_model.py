import importlib.util
import unittest
import tempfile
from pathlib import Path

from openprop.comparators import default_comparators
from openprop.matcher import EntityMatcher
from openprop.models import (
    Entity,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    QueryFrame,
    RelationValue,
    ValueType,
)
from openprop.property_registry import PropertyRegistry
from openprop.selectors import MentionBasedSelector
from openprop.temporal import FreshnessResult


class ConstantPersistenceModel:
    def predict(self, definition, observation, entity, *, as_of):
        return FreshnessResult(0.25, as_of - observation.timestamp, 0.25, 1.0)


class PersistenceModelTests(unittest.TestCase):
    def test_matcher_accepts_injected_persistence_model(self):
        registry = PropertyRegistry()
        registry.register(PropertyDefinition("color", "color", ValueType.CATEGORICAL))
        entity = Entity("cup", {"color": Observation("red", timestamp=0)})
        query = QueryFrame("red", (PropertyConstraint("color", "red"),))
        matcher = EntityMatcher(
            registry,
            default_comparators(),
            MentionBasedSelector(),
            persistence_model=ConstantPersistenceModel(),
        )
        result = matcher.match(query, [entity], as_of=10)[0]
        self.assertEqual(result.match_score, 1.0)
        self.assertEqual(result.coverage, 0.25)
        self.assertEqual(result.score, 0.25)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_synthetic_data_contains_events_and_censoring(self):
        from openprop.synthetic_persistence import contextual_location_data

        examples = contextual_location_data(samples_per_context=50)
        self.assertTrue(any(example.event_observed for example in examples))
        self.assertTrue(any(not example.event_observed for example in examples))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_neural_model_learns_cabinet_state_is_more_persistent(self):
        from openprop.neural_persistence import NeuralPersistenceModel
        from openprop.synthetic_persistence import contextual_location_data

        data = contextual_location_data(samples_per_context=60)
        training = NeuralPersistenceModel.fit(
            data,
            epochs=60,
            hidden_dim=12,
            depth=2,
        )
        model = training.model
        table = model.survival_probability(
            property_name="location",
            subject_type="cup",
            state_predicate="on",
            context_object="table",
            scene="kitchen",
            duration_seconds=5 * 3600,
        )
        cabinet = model.survival_probability(
            property_name="location",
            subject_type="cup",
            state_predicate="inside",
            context_object="cabinet",
            scene="kitchen",
            duration_seconds=5 * 3600,
        )
        self.assertLess(training.final_loss, training.initial_loss)
        self.assertGreater(cabinet, table + 0.4)
        self.assertGreater(model.calibrate(data), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            restored = NeuralPersistenceModel.load(path)
            self.assertAlmostEqual(
                restored.hazard_per_hour(("location", "cup", "inside", "cabinet", "kitchen")),
                model.hazard_per_hour(("location", "cup", "inside", "cabinet", "kitchen")),
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
