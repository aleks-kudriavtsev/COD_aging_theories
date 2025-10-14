"""Ontology-aware search query generation utilities.

This component will turn ontology nodes into actionable search strings using the
LLM prompt contract defined in the architecture. It will produce boolean queries
and curated term lists that feed back into OpenAlex, PubMed, and other API
integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QueryBundle:
    """Represents a set of search assets aligned to an ontology node."""

    boolean_queries: List[str]
    positive_terms: List[str]
    negative_terms: List[str]
    organism_terms: List[str]
    method_terms: List[str]


class QueryGenerator:
    """Placeholder LLM-driven query generator."""

    def generate(self, node: Dict[str, str]) -> QueryBundle:
        """Generate search assets for a single ontology node.

        Final implementations will call out to LLM services, incorporate curated
        vocabularies, and log prompts/outputs for reproducibility. Currently
        returns empty query placeholders.
        """

        return QueryBundle(
            boolean_queries=[],
            positive_terms=[],
            negative_terms=[],
            organism_terms=[],
            method_terms=[],
        )
