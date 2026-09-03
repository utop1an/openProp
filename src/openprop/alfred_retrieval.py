from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .alfred_adapter import AlfredLanguageCase
from .alfred_ontology import normalise_label
from .models import QueryFrame


@dataclass(frozen=True, slots=True)
class AlfredRetrievalResult:
    training_case_id: str
    training_task_id: str
    score: float
    frame: QueryFrame


class AlfredBM25FrameRetriever:
    """Deterministic train-only sparse retrieval baseline for typed goal frames."""

    def __init__(
        self,
        cases: Iterable[AlfredLanguageCase],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        rows = tuple(cases)
        if not rows:
            raise ValueError("at least one ALFRED training case is required")
        if any(case.split != "train" for case in rows):
            raise ValueError("BM25 frame retriever accepts train-split cases only")
        case_ids = [case.case_id for case in rows]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("BM25 frame retriever requires unique case IDs")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.cases = rows
        self.k1 = float(k1)
        self.b = float(b)
        self._term_counts = [Counter(self._tokens(case.query)) for case in rows]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        if not all(self._lengths):
            raise ValueError("training queries cannot be empty")
        self._average_length = sum(self._lengths) / len(self._lengths)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, counts in enumerate(self._term_counts):
            for term, frequency in counts.items():
                postings[term].append((index, frequency))
        self._postings = dict(postings)
        document_count = len(rows)
        self._idf = {
            term: math.log(
                1.0 + (document_count - len(entries) + 0.5) / (len(entries) + 0.5)
            )
            for term, entries in self._postings.items()
        }

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(normalise_label(text).split())

    def retrieve(self, query: str, *, limit: int = 1) -> tuple[AlfredRetrievalResult, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query_counts = Counter(self._tokens(query))
        if not query_counts:
            raise ValueError("query text cannot be empty")
        scores: dict[int, float] = defaultdict(float)
        for term, query_frequency in query_counts.items():
            entries = self._postings.get(term)
            if entries is None:
                continue
            idf = self._idf[term]
            for index, term_frequency in entries:
                length_ratio = self._lengths[index] / self._average_length
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[index] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_frequency
                )
        ranked = sorted(
            scores,
            key=lambda index: (-scores[index], self.cases[index].case_id),
        )[:limit]
        return tuple(
            AlfredRetrievalResult(
                self.cases[index].case_id,
                self.cases[index].task_id,
                scores[index],
                QueryFrame(query, self.cases[index].gold_frame.constraints),
            )
            for index in ranked
        )

    def audit(self) -> dict[str, object]:
        return {
            "source_split": "train",
            "documents": len(self.cases),
            "unique_tasks": len({case.task_id for case in self.cases}),
            "vocabulary_terms": len(self._postings),
            "k1": self.k1,
            "b": self.b,
            "validation_data_used_for_fit": False,
            "no_shared_term_policy": "return no result",
            "tie_break": "training case ID lexical order",
        }
