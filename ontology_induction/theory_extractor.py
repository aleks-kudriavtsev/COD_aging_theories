"""Candidate theory extraction from parsed review documents.

Implements the prompts and heuristics necessary to surface aging theory labels,
synonyms, and evidence snippets from structured text. Final versions will rely
on spaCy/scispaCy, YAKE/KeyBERT, and LLM calls, emitting JSON objects compatible
with the ontology schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class TheoryCandidate:
    """Represents a potential aging theory extracted from a document."""

    label: str
    synonyms: List[str]
    snippet: str
    confidence: float


class TheoryExtractor:
    """Stub for the ontology induction stage."""

    def extract(self, documents: Iterable[dict]) -> List[TheoryCandidate]:
        """Extract candidate theories from parsed review documents.

        When implemented, this method will call configured LLM prompts, apply
        term-mining heuristics, and gather evidence snippets to seed the initial
        ontology graph. Currently, it returns an empty list placeholder.
        """

        return []
