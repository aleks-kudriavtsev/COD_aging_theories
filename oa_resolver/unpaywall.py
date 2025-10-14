"""Resolution of lawful open-access full texts via the Unpaywall ecosystem.

The final implementation will coordinate Unpaywall lookups with PubMed Central,
CORE, and OpenAIRE fallbacks before permitting downloads for parsing. It should
persist license evidence and preferred URLs so the Prefect/make orchestrator can
ensure compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class OARecord:
    """Represents resolved OA metadata for a document placeholder."""

    doi: str
    license: Optional[str] = None
    best_url: Optional[str] = None
    evidence: Optional[str] = None


class UnpaywallResolver:
    """Stub for integrating Unpaywall, PMC, CORE, and OpenAIRE."""

    def resolve(self, dois: Iterable[str]) -> List[OARecord]:
        """Return OA metadata for the provided DOIs.

        The concrete implementation will invoke Unpaywall's API, check PMC and
        other OA hubs when Unpaywall yields no direct PDF, and retain provenance
        for later auditing. For now, this stub simply returns an empty list.
        """

        return []
