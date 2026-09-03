import unittest
from dataclasses import replace

from openprop.alfred_adapter import AlfredLanguageCase
from openprop.alfred_retrieval import AlfredBM25FrameRetriever
from openprop.models import PropertyConstraint, QueryFrame, RelationValue


def _case(case_id, query, constraints, *, split="train"):
    return AlfredLanguageCase(
        case_id,
        case_id,
        split,
        "pick_and_place_simple",
        "FloorPlan1",
        query,
        QueryFrame(query, tuple(constraints)),
        0,
        f"{split}/{case_id}/traj_data.json",
    )


class AlfredRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.apple = _case(
            "a-apple",
            "Put the apple in the fridge.",
            (
                PropertyConstraint("type", "apple", 0.5),
                PropertyConstraint(
                    "location", RelationValue("inside", {"object": "fridge"}), 0.5
                ),
            ),
        )
        self.mug = _case(
            "b-mug",
            "Wash the mug and place it in the sink.",
            (
                PropertyConstraint("type", "mug", 0.35),
                PropertyConstraint("cleanliness", "clean", 0.35),
                PropertyConstraint(
                    "location", RelationValue("inside", {"object": "sink basin"}), 0.3
                ),
            ),
        )

    def test_exact_query_retrieves_typed_training_frame(self):
        result = AlfredBM25FrameRetriever((self.apple, self.mug)).retrieve(
            self.apple.query
        )[0]
        self.assertEqual("a-apple", result.training_case_id)
        self.assertEqual(self.apple.gold_frame.constraints, result.frame.constraints)
        self.assertEqual(self.apple.query, result.frame.text)
        self.assertIsInstance(result.frame.constraints[1].desired_value, RelationValue)

    def test_no_shared_term_returns_no_result(self):
        retriever = AlfredBM25FrameRetriever((self.apple, self.mug))
        self.assertEqual((), retriever.retrieve("xyzzy plugh"))

    def test_ties_are_deterministic_by_case_id(self):
        first = replace(self.apple, case_id="a", task_id="a", query="move object")
        second = replace(self.apple, case_id="b", task_id="b", query="move object")
        retriever = AlfredBM25FrameRetriever((second, first))
        self.assertEqual(
            ["a", "b"],
            [item.training_case_id for item in retriever.retrieve("move object", limit=2)],
        )

    def test_rejects_nontrain_cases_duplicate_ids_and_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "train-split"):
            AlfredBM25FrameRetriever((replace(self.apple, split="valid_unseen"),))
        with self.assertRaisesRegex(ValueError, "unique case IDs"):
            AlfredBM25FrameRetriever((self.apple, self.apple))
        with self.assertRaisesRegex(ValueError, "k1"):
            AlfredBM25FrameRetriever((self.apple,), k1=0)
        with self.assertRaisesRegex(ValueError, "between"):
            AlfredBM25FrameRetriever((self.apple,), b=1.1)

    def test_audit_declares_train_only_fit_and_no_evidence_policy(self):
        audit = AlfredBM25FrameRetriever((self.apple, self.mug)).audit()
        self.assertEqual("train", audit["source_split"])
        self.assertFalse(audit["validation_data_used_for_fit"])
        self.assertEqual("return no result", audit["no_shared_term_policy"])


if __name__ == "__main__":
    unittest.main()
