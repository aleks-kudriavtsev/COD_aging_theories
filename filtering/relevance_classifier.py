"""Relevance classification scaffold for aging theory detection.

This module combines heuristic token checks with an OpenAI LLM confirmation
step to decide whether a record discusses an aging theory. Outputs are
structured JSON-like decisions with confidence and rationales for auditing.
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
    from openai import OpenAI
    try:  # OpenAIError lives in different modules depending on package version
        from openai import OpenAIError  # type: ignore
    except Exception:  # pragma: no cover
        from openai.error import OpenAIError  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore


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
    """Two-stage classifier that combines heuristics with an LLM confirmation."""

    def __init__(
        self,
        log_path: Optional[Path | str] = None,
        openai_model: str = "gpt-4o-mini",
        acceptance_threshold: float = 0.65,
        uncertainty_margin: float = 0.55,
    ) -> None:
        self.log_path = Path(log_path or "logs/corpus_seed.jsonl")
        self.openai_model = openai_model
        self.acceptance_threshold = acceptance_threshold
        self.uncertainty_margin = uncertainty_margin
        self._openai_client: Optional[OpenAI] = None
        self.summary = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "uncertain": 0,
            "heuristic_rejects": 0,
            "llm_rejects": 0,
        }
        self.rejection_reasons: Counter[str] = Counter()

    def _ensure_openai_client(self) -> Optional[OpenAI]:
        """Initialise an OpenAI client if the dependency is installed."""

        if self._openai_client is not None:
            return self._openai_client

        if OpenAI is None:
            LOGGER.warning(
                "openai package is unavailable; classification will rely on heuristics only."
            )
            return None

        try:
            self._openai_client = OpenAI()
        except Exception as exc:  # pragma: no cover - runtime errors
            LOGGER.warning("Failed to initialise OpenAI client: %s", exc)
            self._openai_client = None
        return self._openai_client

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

    def _call_openai_classifier(self, record: dict, heuristic_terms: List[str]) -> Optional[dict]:
        """Send a structured relevance prompt to an OpenAI LLM."""

        client = self._ensure_openai_client()
        if client is None:
            return None

        title = (record.get("title") or "").strip()
        abstract = (record.get("abstract") or "").strip()

        if not (title or abstract):
            LOGGER.debug("Skipping LLM stage because the record lacks title and abstract")
            return None

        key_term_fragment = ", ".join(heuristic_terms) if heuristic_terms else "(none)"
        user_prompt = (
            "You are reviewing scientific records about theories of biological aging.\n"
            "Decide whether the document discusses conceptual models, hypotheses, or formal theories of aging.\n"
            "Reject documents focused solely on specific mechanisms, biomarkers, or single tissues without broader theoretical framing.\n"
            "Return a strict JSON object with keys decision (true/false/uncertain), confidence (0-1 float), rationale (one sentence), and key_terms (list of short phrases).\n"
            "Use uncertainty when evidence is ambiguous.\n"
            "Title: "
            f"{title or 'N/A'}\n"
            "Abstract: "
            f"{abstract or 'N/A'}\n"
            "Heuristic terms observed: {key_term_fragment}"
        )

        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.openai_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful research assistant."
                            " Respond only with valid JSON for downstream ingestion."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AttributeError:
            try:
                completion = client.responses.create(  # type: ignore[attr-defined]
                    model=self.openai_model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful research assistant."
                                " Respond only with valid JSON for downstream ingestion."
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:  # pragma: no cover - runtime/API errors
                LOGGER.warning("OpenAI responses endpoint failed: %s", exc)
                return None

            if hasattr(completion, "output_text") and completion.output_text:  # type: ignore[attr-defined]
                message_content = completion.output_text  # type: ignore[assignment]
            else:
                output = completion.output if hasattr(completion, "output") else None
                if not output:
                    LOGGER.warning("OpenAI responses endpoint returned empty output")
                    return None
                try:
                    message_content = output[0]["content"][0]["text"]  # type: ignore[index]
                except Exception as exc:  # pragma: no cover - schema drift
                    LOGGER.warning("Unexpected responses payload: %s", exc)
                    return None
        except OpenAIError as exc:  # pragma: no cover - API errors
            LOGGER.warning("OpenAI classification request failed: %s", exc)
            return None
        else:
            try:
                message_content = response.choices[0].message.content  # type: ignore[index]
            except Exception as exc:  # pragma: no cover - schema drift
                LOGGER.warning("Unexpected chat completion payload: %s", exc)
                return None

        if not message_content:
            LOGGER.warning("OpenAI classification returned empty message content")
            return None

        try:
            parsed = json.loads(message_content)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse OpenAI JSON payload: %s", exc)
            return None

        return parsed

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

            llm_payload = None
            decision = "false"
            confidence = 0.0
            rationale_parts = []
            reason: Optional[str] = None

            if heuristic.passed:
                rationale_parts.append(
                    "Heuristic tokens detected: " + ", ".join(heuristic.key_terms)
                )
                llm_payload = self._call_openai_classifier(record, heuristic.key_terms)
                if llm_payload is None or not isinstance(llm_payload, dict):
                    decision = "uncertain"
                    confidence = 0.5
                    rationale_parts.append(
                        "OpenAI classifier unavailable or returned invalid payload; keeping record for manual review."
                    )
                    self.summary["uncertain"] += 1
                    reason = "llm_unavailable" if llm_payload is None else "llm_invalid_payload"
                else:
                    decision_raw = str(llm_payload.get("decision", "")).strip().lower()
                    try:
                        confidence = float(llm_payload.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        confidence = 0.0
                    rationale_text = str(llm_payload.get("rationale", "")).strip()
                    if rationale_text:
                        rationale_parts.append(rationale_text)
                    llm_terms = llm_payload.get("key_terms") or []
                    if isinstance(llm_terms, list):
                        for term in llm_terms:
                            if isinstance(term, str) and term not in heuristic.key_terms:
                                heuristic.key_terms.append(term)
                    heuristic.key_terms = list(dict.fromkeys(heuristic.key_terms))

                    confidence = max(0.0, min(1.0, confidence))

                    if decision_raw == "true" and confidence >= self.acceptance_threshold:
                        decision = "true"
                        rationale_parts.append(
                            f"OpenAI classifier predicted relevance with confidence {confidence:.2f}."
                        )
                        self.summary["accepted"] += 1
                    elif decision_raw == "false" and confidence >= self.uncertainty_margin:
                        decision = "false"
                        rationale_parts.append(
                            f"OpenAI classifier predicted non-relevance with confidence {confidence:.2f}."
                        )
                        self.summary["rejected"] += 1
                        self.summary["llm_rejects"] += 1
                        reason = "llm_not_relevant"
                    elif confidence < self.uncertainty_margin:
                        decision = "uncertain"
                        rationale_parts.append(
                            f"Classifier confidence {confidence:.2f} below uncertainty margin."
                        )
                        self.summary["uncertain"] += 1
                        reason = "llm_low_confidence"
                    elif decision_raw == "true":
                        decision = "uncertain"
                        rationale_parts.append(
                            f"Classifier marked relevant but score {confidence:.2f} below acceptance threshold {self.acceptance_threshold:.2f}."
                        )
                        self.summary["uncertain"] += 1
                        reason = "llm_below_acceptance"
                    elif decision_raw == "false":
                        decision = "false"
                        rationale_parts.append(
                            f"OpenAI classifier predicted non-relevance with confidence {confidence:.2f}."
                        )
                        self.summary["rejected"] += 1
                        self.summary["llm_rejects"] += 1
                        reason = "llm_not_relevant"
                    elif decision_raw == "uncertain":
                        decision = "uncertain"
                        rationale_parts.append("OpenAI classifier flagged the record as uncertain.")
                        self.summary["uncertain"] += 1
                        reason = "llm_flagged_uncertain"
                    else:
                        # Any unexpected label is treated as uncertain for safety.
                        decision = "uncertain"
                        rationale_parts.append(
                            f"Received unexpected decision '{decision_raw}'; marking as uncertain."
                        )
                        self.summary["uncertain"] += 1
                        reason = "llm_unexpected_decision"
            else:
                decision = "false"
                confidence = 0.2
                missing_parts = ", ".join(heuristic.failure_reasons)
                rationale_parts.append(
                    f"Failed heuristic checks ({missing_parts}); skipping LLM stage."
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

            if llm_payload is not None:
                sanitized_payload = {
                    "decision": llm_payload.get("decision"),
                    "confidence": llm_payload.get("confidence"),
                    "rationale": llm_payload.get("rationale"),
                    "key_terms": llm_payload.get("key_terms"),
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
                    "llm_payload": sanitized_payload,
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
