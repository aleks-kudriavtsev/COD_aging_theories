"""Relevance classification scaffold for aging theory detection.

This module will combine heuristic token checks with ML or LLM classifiers such
as cross-encoder/ms-marco-MiniLM-L-6-v2 to decide whether a record discusses an
aging theory. Outputs are structured JSON-like decisions with confidence and
rationales for auditing.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from transformers import pipeline
except Exception:  # pragma: no cover - graceful degradation
    pipeline = None  # type: ignore


LOGGER = logging.getLogger(__name__)

AGING_PATTERN = re.compile(r"\b(ageing?|aging|senescen\w*)\b", re.IGNORECASE)
THEORY_PATTERN = re.compile(r"\b(theor\w*|model\w*|hypothes\w*)\b", re.IGNORECASE)


def _extract_text(record: dict) -> str:
    """Concatenate title and abstract for downstream checks."""

    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    return " ".join(part for part in (title, abstract) if part).strip()


def _collect_terms(pattern: re.Pattern[str], text: str) -> List[str]:
    """Return lowercased, de-duplicated matches for a regex pattern."""

    matches = pattern.findall(text)
    # `findall` may return tuples if the pattern has capturing groups; flatten
    if matches and isinstance(matches[0], tuple):
        flat = []
        for match in matches:
            flat.extend([piece for piece in match if piece])
        matches = flat
    return list(dict.fromkeys(token.lower() for token in matches if token))


@dataclass
class ClassificationResult:
    """Represents the relevance decision for a single document."""

    decision: str
    confidence: float
    rationale: str
    key_terms: List[str]


@dataclass
class _HeuristicOutcome:
    """Internal representation of heuristic filtering results."""

    passed: bool
    key_terms: List[str]
    failure_reasons: List[str] = field(default_factory=list)


class RelevanceClassifier:
    """Two-stage classifier that combines heuristics with an ML confirmation."""

    def __init__(
        self,
        log_path: Optional[Path | str] = None,
        zero_shot_model: str = "facebook/bart-large-mnli",
        acceptance_threshold: float = 0.65,
        uncertainty_margin: float = 0.55,
    ) -> None:
        self.log_path = Path(log_path or "logs/corpus_seed.jsonl")
        self.zero_shot_model = zero_shot_model
        self.acceptance_threshold = acceptance_threshold
        self.uncertainty_margin = uncertainty_margin
        self._zero_shot = None
        self.summary = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "uncertain": 0,
            "heuristic_rejects": 0,
            "ml_rejects": 0,
        }
        self.rejection_reasons: Counter[str] = Counter()

    def _ensure_zero_shot(self) -> Optional[object]:
        """Load a zero-shot classifier if the dependency is available."""

        if self._zero_shot is not None:
            return self._zero_shot

        if pipeline is None:
            LOGGER.warning(
                "transformers pipeline is unavailable; classification will rely on heuristics only."
            )
            return None

        try:
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model=self.zero_shot_model,
                truncation=True,
            )
        except Exception as exc:  # pragma: no cover - download/runtime errors
            LOGGER.warning("Failed to initialise zero-shot pipeline: %s", exc)
            self._zero_shot = None
        return self._zero_shot

    def _heuristic_pass(self, record: dict) -> _HeuristicOutcome:
        """Apply keyword heuristics to detect likely aging-theory documents."""

        text = _extract_text(record)
        lowered = text.lower()
        aging_terms = _collect_terms(AGING_PATTERN, lowered)
        theory_terms = _collect_terms(THEORY_PATTERN, lowered)

        failure_reasons: List[str] = []
        if not aging_terms:
            failure_reasons.append("missing_aging_terms")
        if not theory_terms:
            failure_reasons.append("missing_theory_terms")

        key_terms = aging_terms + [term for term in theory_terms if term not in aging_terms]
        passed = not failure_reasons
        return _HeuristicOutcome(passed=passed, key_terms=key_terms, failure_reasons=failure_reasons)

    def _run_zero_shot(self, text: str) -> Optional[dict]:
        """Execute zero-shot classification when the model is available."""

        if not text:
            return None

        classifier = self._ensure_zero_shot()
        if classifier is None:
            return None

        return classifier(
            text,
            candidate_labels=["relevant", "not relevant"],
            hypothesis_template="This document is {} to aging theory research.",
        )

    def _log_decision(self, entry: dict) -> None:
        """Persist a structured decision log for downstream ingest steps."""

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def classify(self, records: Iterable[dict]) -> List[ClassificationResult]:
        """Classify incoming records for their focus on aging theory."""

        results: List[ClassificationResult] = []
        for record in records:
            self.summary["total"] += 1
            heuristic = self._heuristic_pass(record)
            text = _extract_text(record)

            ml_payload = None
            decision = "false"
            confidence = 0.0
            rationale_parts = []
            reason: Optional[str] = None

            if heuristic.passed:
                rationale_parts.append(
                    "Heuristic tokens detected: " + ", ".join(heuristic.key_terms)
                )
                ml_payload = self._run_zero_shot(text)
                if ml_payload is None:
                    decision = "uncertain"
                    confidence = 0.5
                    rationale_parts.append(
                        "Zero-shot classifier unavailable; keeping record for manual review."
                    )
                    self.summary["uncertain"] += 1
                    reason = "ml_unavailable"
                else:
                    top_label = ml_payload["labels"][0]
                    top_score = float(ml_payload["scores"][0])
                    if top_label == "relevant" and top_score >= self.acceptance_threshold:
                        decision = "true"
                        confidence = top_score
                        rationale_parts.append(
                            f"Zero-shot classifier predicted relevance with score {top_score:.2f}."
                        )
                        self.summary["accepted"] += 1
                    elif top_score < self.uncertainty_margin:
                        decision = "uncertain"
                        confidence = top_score
                        rationale_parts.append(
                            f"Classifier confidence {top_score:.2f} below uncertainty margin."
                        )
                        self.summary["uncertain"] += 1
                        reason = "ml_low_confidence"
                    else:
                        decision = "false"
                        confidence = top_score
                        rationale_parts.append(
                            f"Zero-shot classifier predicted non-relevance with score {top_score:.2f}."
                        )
                        self.summary["rejected"] += 1
                        self.summary["ml_rejects"] += 1
                        reason = "ml_not_relevant"
            else:
                decision = "false"
                confidence = 0.2
                missing_parts = ", ".join(heuristic.failure_reasons)
                rationale_parts.append(
                    f"Failed heuristic checks ({missing_parts}); skipping ML stage."
                )
                self.summary["rejected"] += 1
                self.summary["heuristic_rejects"] += 1
                reason = "heuristic_failure"

            if reason:
                self.rejection_reasons[reason] += 1

            result = ClassificationResult(
                decision=decision,
                confidence=confidence,
                rationale=" ".join(rationale_parts),
                key_terms=heuristic.key_terms,
            )
            results.append(result)

            if ml_payload is not None:
                sanitized_payload = {
                    "labels": list(ml_payload.get("labels", [])),
                    "scores": [float(score) for score in ml_payload.get("scores", [])],
                    "sequence": ml_payload.get("sequence"),
                }
            else:
                sanitized_payload = None

            log_entry = {
                "record": record,
                "classification": {
                    "decision": decision,
                    "confidence": confidence,
                    "rationale": result.rationale,
                    "key_terms": heuristic.key_terms,
                    "heuristic": {
                        "passed": heuristic.passed,
                        "failure_reasons": heuristic.failure_reasons,
                    },
                    "ml_payload": sanitized_payload,
                    "rejection_reason": reason,
                },
            }
            self._log_decision(log_entry)

        return results

    def summary_statistics(self) -> dict:
        """Return a dictionary with aggregate decision statistics."""

        stats = dict(self.summary)
        stats["rejection_reasons"] = dict(self.rejection_reasons)
        return stats
