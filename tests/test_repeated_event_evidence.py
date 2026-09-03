from __future__ import annotations

import unittest

from openprop.persistence_data import PersistenceTrainingExample
from openprop.repeated_event_evidence import (
    RepeatedEventEvidence,
    decode_repeated_event_consensus,
    estimate_independent_symmetric_noise,
    simulate_repeated_event_evidence,
    single_annotation_examples,
)


def _rows(count: int = 100) -> tuple[PersistenceTrainingExample, ...]:
    return tuple(
        PersistenceTrainingExample(
            property_name="location",
            subject_type="cup" if index % 2 else "book",
            state_predicate="inside",
            context_object="drawer",
            scene="busy" if index % 3 else "quiet",
            duration_seconds=300.0 + index,
            event_observed=index % 4 != 0,
            group_id=f"case-{index:03d}",
        )
        for index in range(count)
    )


class RepeatedEventEvidenceTests(unittest.TestCase):
    def test_simulation_is_deterministic_and_does_not_expose_latent_truth(self):
        rows = _rows()
        left = simulate_repeated_event_evidence(
            rows, annotator_count=5, flip_fraction=0.2, seed=17
        )
        right = simulate_repeated_event_evidence(
            rows, annotator_count=5, flip_fraction=0.2, seed=17
        )
        self.assertEqual(left, right)
        self.assertFalse(hasattr(left[0], "event_observed"))
        for index in range(5):
            errors = sum(
                row.event_labels[index] != truth.event_observed
                for row, truth in zip(left, rows, strict=True)
            )
            self.assertEqual(20, errors)

    def test_noise_estimate_uses_pairwise_disagreement_identity(self):
        evidence = simulate_repeated_event_evidence(
            _rows(1000), annotator_count=5, flip_fraction=0.2, seed=23
        )
        estimate = estimate_independent_symmetric_noise(evidence)
        self.assertEqual(10_000, estimate.compared_annotation_pairs)
        self.assertAlmostEqual(0.2, estimate.flip_probability, delta=0.02)
        self.assertTrue(estimate.identifiable)

    def test_consensus_reduces_label_errors_and_records_budget(self):
        truth = _rows(1000)
        evidence = simulate_repeated_event_evidence(
            truth, annotator_count=5, flip_fraction=0.2, seed=29
        )
        single = single_annotation_examples(evidence)
        consensus = decode_repeated_event_consensus(evidence)
        single_errors = sum(
            row.event_observed != expected.event_observed
            for row, expected in zip(single, truth, strict=True)
        )
        consensus_errors = sum(
            row.event_observed != expected.event_observed
            for row, expected in zip(consensus.examples, truth, strict=True)
        )
        self.assertEqual(200, single_errors)
        self.assertLess(consensus_errors, single_errors)
        self.assertEqual(5000, consensus.annotation_budget)
        self.assertEqual(1000, consensus.retained_records)
        self.assertEqual(0, consensus.abstained_records)

    def test_confidence_threshold_abstains_on_weak_majority(self):
        evidence = (
            RepeatedEventEvidence(
                "location",
                "cup",
                "inside",
                "drawer",
                "busy",
                600.0,
                "weak",
                ("a", "b", "c", "d", "e"),
                (True, True, True, False, False),
            ),
            RepeatedEventEvidence(
                "location",
                "cup",
                "inside",
                "drawer",
                "busy",
                700.0,
                "strong",
                ("a", "b", "c", "d", "e"),
                (True, True, True, True, True),
            ),
        )
        consensus = decode_repeated_event_consensus(
            evidence, minimum_posterior_confidence=0.9
        )
        self.assertEqual(("strong",), tuple(row.group_id for row in consensus.examples))
        self.assertEqual(1, consensus.abstained_records)

    def test_invalid_or_unidentifiable_evidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "odd number"):
            RepeatedEventEvidence(
                "location",
                "cup",
                "inside",
                "drawer",
                "busy",
                1.0,
                "x",
                ("a", "b"),
                (True, False),
            )
        chance = (
            RepeatedEventEvidence(
                "location",
                "cup",
                "inside",
                "drawer",
                "busy",
                float(index + 1),
                f"chance-{index}",
                ("a", "b", "c"),
                labels,
            )
            for index, labels in enumerate(
                ((True, True, False), (False, False, True), (True, False, True))
            )
        )
        with self.assertRaisesRegex(ValueError, "chance-level"):
            decode_repeated_event_consensus(chance)


if __name__ == "__main__":
    unittest.main()
