from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Entity, Observation, PropertyDefinition, RelationValue
from .persistence import ExponentialPersistenceModel
from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import exponential_example_negative_log_likelihood
from .temporal import FreshnessResult

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover - exercised when optional extra is absent.
    raise ImportError(
        "Neural persistence requires the 'ml' extra: python -m pip install -e .[ml]"
    ) from error


FEATURE_NAMES = (
    "property_name",
    "subject_type",
    "state_predicate",
    "context_object",
    "scene",
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: "NeuralPersistenceModel"
    initial_loss: float
    final_loss: float
    epochs: int


class _Vocabulary:
    def __init__(self, values: Iterable[str]) -> None:
        unique = sorted({value.casefold() for value in values if value})
        self.tokens = ("<unk>", *unique)
        self.indices = {token: index for index, token in enumerate(self.tokens)}

    def encode(self, value: str) -> int:
        return self.indices.get(value.casefold(), 0)


class _HazardNetwork(nn.Module):
    def __init__(
        self,
        vocabulary_sizes: tuple[int, ...],
        *,
        embedding_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.vocabulary_sizes = vocabulary_sizes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in vocabulary_sizes
        )
        layers: list[nn.Module] = []
        width = embedding_dim * len(vocabulary_sizes)
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.ReLU()))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = [
            embedding(features[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return F.softplus(self.layers(torch.cat(encoded, dim=1)).squeeze(1)) + 1e-6


class NeuralPersistenceModel:
    """Context-conditioned exponential hazard predicted by a neural network."""

    def __init__(
        self,
        network: _HazardNetwork,
        vocabularies: tuple[_Vocabulary, ...],
        trained_properties: frozenset[str],
        hazard_scale: float = 1.0,
    ) -> None:
        self.network = network.eval()
        self.vocabularies = vocabularies
        self.trained_properties = trained_properties
        self.hazard_scale = hazard_scale
        self.fallback = ExponentialPersistenceModel()

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
        *,
        epochs: int = 300,
        learning_rate: float = 0.02,
        embedding_dim: int = 6,
        hidden_dim: int = 24,
        depth: int = 2,
        seed: int = 7,
    ) -> TrainingResult:
        data = tuple(examples)
        if not data:
            raise ValueError("at least one training example is required")
        if epochs <= 0 or depth <= 0:
            raise ValueError("epochs and depth must be positive")
        torch.manual_seed(seed)
        columns = tuple(zip(*(example.features() for example in data), strict=True))
        vocabularies = tuple(_Vocabulary(column) for column in columns)
        features = torch.tensor(
            [
                [vocabulary.encode(value) for vocabulary, value in zip(vocabularies, example.features(), strict=True)]
                for example in data
            ],
            dtype=torch.long,
        )
        durations = torch.tensor(
            [example.duration_seconds / 3600.0 for example in data],
            dtype=torch.float32,
        )
        observed = torch.tensor(
            [float(example.event_observed) for example in data],
            dtype=torch.float32,
        )
        interval_lower = torch.tensor(
            [
                -1.0
                if example.interval_start_seconds is None
                else example.interval_start_seconds / 3600.0
                for example in data
            ],
            dtype=torch.float32,
        )
        network = _HazardNetwork(
            tuple(len(vocabulary.tokens) for vocabulary in vocabularies),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        initial_loss = 0.0
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            hazard = network(features)
            # Exact events, right censoring, and inspection intervals each use
            # their corresponding exponential likelihood contribution.
            exact_or_censored = hazard * durations - observed * torch.log(hazard)
            interval_mask = interval_lower >= 0
            interval_width = torch.clamp(durations - interval_lower, min=1e-12)
            interval_probability = torch.clamp(
                -torch.expm1(-hazard * interval_width), min=1e-12
            )
            interval_loss = hazard * interval_lower - torch.log(interval_probability)
            loss = torch.where(interval_mask, interval_loss, exact_or_censored).mean()
            if epoch == 0:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

        model = cls(
            network,
            vocabularies,
            frozenset(example.property_name.casefold() for example in data),
        )
        return TrainingResult(model, initial_loss, final_loss, epochs)

    def calibrate(self, examples: Iterable[PersistenceTrainingExample]) -> float:
        """Fit a global hazard multiplier on a held-out validation split."""
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one calibration example is required")
        if any(example.is_interval_censored for example in rows):
            current_scale = self.hazard_scale or 1.0
            base_hazards = [
                self.hazard_per_hour(example.features()) / current_scale
                for example in rows
            ]

            def objective(log_scale: float) -> float:
                scale = math.exp(log_scale)
                return sum(
                    exponential_example_negative_log_likelihood(base * scale, example)
                    for base, example in zip(base_hazards, rows, strict=True)
                ) / len(rows)

            left, right = -8.0, 8.0
            ratio = (math.sqrt(5.0) - 1.0) / 2.0
            for _ in range(100):
                x1 = right - ratio * (right - left)
                x2 = left + ratio * (right - left)
                if objective(x1) <= objective(x2):
                    right = x2
                else:
                    left = x1
            self.hazard_scale = max(
                0.01, min(100.0, math.exp((left + right) / 2.0))
            )
            return self.hazard_scale
        event_count = sum(example.event_observed for example in rows)
        denominator = 0.0
        current_scale = self.hazard_scale or 1.0
        for example in rows:
            base_hazard = self.hazard_per_hour(example.features()) / current_scale
            denominator += base_hazard * example.duration_seconds / 3600.0
        if event_count == 0 or denominator <= 0:
            raise ValueError("calibration requires observed transitions and positive exposure")
        self.hazard_scale = max(0.01, min(100.0, event_count / denominator))
        return self.hazard_scale

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "feature_names": FEATURE_NAMES,
                "network": {
                    "vocabulary_sizes": self.network.vocabulary_sizes,
                    "embedding_dim": self.network.embedding_dim,
                    "hidden_dim": self.network.hidden_dim,
                    "depth": self.network.depth,
                },
                "vocabularies": [vocabulary.tokens for vocabulary in self.vocabularies],
                "trained_properties": sorted(self.trained_properties),
                "hazard_scale": self.hazard_scale,
                "state_dict": self.network.state_dict(),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPersistenceModel":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported persistence model format")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("persistence model feature schema differs")
        config = payload["network"]
        vocabularies = tuple(
            _Vocabulary(tokens[1:]) for tokens in payload["vocabularies"]
        )
        network = _HazardNetwork(
            tuple(config["vocabulary_sizes"]),
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
        )
        network.load_state_dict(payload["state_dict"])
        return cls(
            network,
            vocabularies,
            frozenset(payload["trained_properties"]),
            float(payload.get("hazard_scale", 1.0)),
        )
    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        encoded = torch.tensor(
            [[vocabulary.encode(value) for vocabulary, value in zip(self.vocabularies, features, strict=True)]],
            dtype=torch.long,
        )
        with torch.no_grad():
            return float(self.network(encoded)[0]) * self.hazard_scale

    def survival_probability(
        self,
        *,
        property_name: str,
        subject_type: str,
        state_predicate: str,
        context_object: str,
        scene: str,
        duration_seconds: float,
    ) -> float:
        hazard = self.hazard_per_hour(
            (property_name, subject_type, state_predicate, context_object, scene)
        )
        return math.exp(-hazard * max(0.0, duration_seconds) / 3600.0)

    def predict(
        self,
        definition: PropertyDefinition,
        observation: Observation,
        entity: Entity,
        *,
        as_of: float,
    ) -> FreshnessResult:
        if (
            observation.timestamp is None
            or definition.name.casefold() not in self.trained_properties
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        age_seconds = max(0.0, as_of - observation.timestamp)
        features = self._features(definition, observation, entity)
        survival = self.survival_probability(
            property_name=features[0],
            subject_type=features[1],
            state_predicate=features[2],
            context_object=features[3],
            scene=features[4],
            duration_seconds=age_seconds,
        )
        # Event effects remain an explicit, auditable layer until event-aware
        # training data is available.
        baseline = self.fallback.predict(definition, observation, entity, as_of=as_of)
        event_retention = baseline.event_retention
        freshness = max(0.0, min(1.0, survival * event_retention))
        return FreshnessResult(
            freshness,
            age_seconds,
            survival,
            event_retention,
            baseline.applied_events,
        )

    @staticmethod
    def _features(
        definition: PropertyDefinition,
        observation: Observation,
        entity: Entity,
    ) -> tuple[str, ...]:
        subject = entity.properties.get("type")
        subject_type = str(subject.value) if subject is not None else "unknown"
        scene_observation = entity.properties.get("scene")
        scene = str(scene_observation.value) if scene_observation is not None else "unknown"
        if isinstance(observation.value, RelationValue):
            predicate = observation.value.predicate
            context_object = str(observation.value.arguments.get("object", "unknown"))
        else:
            predicate = str(observation.value)
            context_object = "none"
        return (definition.name, subject_type, predicate, context_object, scene)
