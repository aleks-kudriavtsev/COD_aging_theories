"""Utilities for harvesting review-focused records from the OpenAlex API."""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

import requests


logger = logging.getLogger(__name__)


def _reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Translate OpenAlex's inverted abstract index into human readable text."""

    if not abstract_inverted_index:
        return None

    # The inverted index maps token -> positions. We invert again by placing
    # tokens at their given positions and joining with spaces.
    positions: Dict[int, str] = {}
    for token, indices in abstract_inverted_index.items():
        for idx in indices:
            positions[idx] = token

    return " ".join(positions[index] for index in sorted(positions)) or None


def _clean_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.lower()
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    return doi


@dataclass
class OpenAlexIngestor:
    """Client for collecting review-oriented works from OpenAlex.

    The ingestor performs authenticated cursor-based pagination, normalizes the
    response into the project's document schema, and optionally enriches sparse
    records with PubMed and Semantic Scholar metadata.
    """

    queries: Iterable[str] = field(
        default_factory=lambda: (
            '"aging theory"',
            '"ageing theory"',
            '"theories of aging"',
            '"senescence theory"',
        )
    )
    per_page: int = 200
    max_records_per_query: Optional[int] = None
    request_timeout: int = 30
    retry_attempts: int = 4
    backoff_seconds: float = 2.0
    rate_limit_interval: float = 0.2
    enrich_missing: bool = True
    session: requests.Session = field(default_factory=requests.Session, init=False)

    def __post_init__(self) -> None:
        headers = {
            "Accept": "application/json",
            "User-Agent": os.getenv("OPENALEX_USER_AGENT", "COD-aging-theories-ingestor/0.1"),
        }
        api_key = os.getenv("OPENALEX_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_reviews(self) -> List[dict]:
        """Fetch review-style records for downstream filtering.

        Returns:
            A list of normalized document dictionaries with OpenAlex provenance
            and optional enrichment information applied.
        """

        documents: List[dict] = []
        for raw_query in self.queries:
            query = raw_query.strip()
            if not query:
                continue
            logger.info(
                "openalex.query.start",
                extra={"query": query, "per_page": self.per_page},
            )

            collected = 0
            for work in self._iter_openalex(query):
                document = self._map_openalex_work(work)
                if self.enrich_missing:
                    document = self._enrich_document(document)
                documents.append(document)
                collected += 1
                if self.max_records_per_query and collected >= self.max_records_per_query:
                    logger.info(
                        "openalex.query.limit_reached",
                        extra={"query": query, "collected": collected},
                    )
                    break

            logger.info(
                "openalex.query.complete",
                extra={"query": query, "records": collected},
            )

        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iter_openalex(self, query: str) -> Iterator[dict]:
        base_url = "https://api.openalex.org/works"
        cursor = "*"
        retrieved = 0

        while True:
            params = {
                "search": query,
                "filter": "type:review,language:en",
                "per-page": self.per_page,
                "cursor": cursor,
            }

            response_json = self._request_with_retry(base_url, params=params)
            results = response_json.get("results", [])
            meta = response_json.get("meta", {})

            for work in results:
                yield work
                retrieved += 1

            cursor = meta.get("next_cursor")
            logger.info(
                "openalex.query.page",
                extra={
                    "query": query,
                    "retrieved": retrieved,
                    "next_cursor": cursor,
                },
            )

            if not cursor:
                break

            if self.rate_limit_interval:
                time.sleep(self.rate_limit_interval)

    def _request_with_retry(self, url: str, params: Dict[str, str]) -> Dict[str, object]:
        mailto = os.getenv("OPENALEX_MAILTO") or os.getenv("UNPAYWALL_EMAIL")
        if mailto:
            params = {**params, "mailto": mailto}

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.request_timeout)
                logger.info(
                    "openalex.request",
                    extra={
                        "url": response.url,
                        "status_code": response.status_code,
                        "attempt": attempt,
                    },
                )
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", self.backoff_seconds))
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt >= self.retry_attempts:
                    logger.error(
                        "openalex.request.failed",
                        extra={"url": url, "params": params, "error": str(exc)},
                    )
                    raise
                sleep_for = self.backoff_seconds * attempt
                logger.warning(
                    "openalex.request.retry",
                    extra={
                        "url": url,
                        "params": params,
                        "error": str(exc),
                        "attempt": attempt,
                        "sleep_for": sleep_for,
                    },
                )
                time.sleep(sleep_for)

        # The loop either returns or raises; this line protects the type checker.
        raise RuntimeError("OpenAlex request retries exhausted")

    def _map_openalex_work(self, work: dict) -> dict:
        doi = _clean_doi(work.get("doi"))
        openalex_id = work.get("id")
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

        authors = []
        for authorship in work.get("authorships", []) or []:
            author = authorship.get("author", {})
            authors.append(
                {
                    "name": author.get("display_name"),
                    "orcid": author.get("orcid") or None,
                }
            )

        concept_entries = []
        for concept in work.get("concepts", []) or []:
            concept_entries.append(
                {
                    "id": concept.get("id"),
                    "display_name": concept.get("display_name"),
                    "level": concept.get("level"),
                    "score": concept.get("score"),
                }
            )

        source_ids = []
        if openalex_id:
            source_ids.append(openalex_id)
        if doi:
            source_ids.append(f"doi:{doi}")

        open_access = work.get("open_access", {}) or {}
        publication_year = work.get("publication_year")
        if publication_year is None and work.get("publication_date"):
            try:
                publication_year = int(work["publication_date"].split("-")[0])
            except (ValueError, AttributeError, IndexError):
                publication_year = None

        provenance = {
            "source": "openalex",
            "openalex_id": openalex_id,
            "retrieved_at": dt.datetime.utcnow().isoformat() + "Z",
            "raw": work,
        }

        document = {
            "id": f"doi:{doi}" if doi else openalex_id,
            "source_ids": source_ids,
            "title": work.get("title"),
            "abstract": abstract,
            "doi": doi,
            "year": publication_year,
            "venue": (work.get("host_venue") or {}).get("display_name"),
            "type": work.get("type"),
            "language": work.get("language"),
            "authors": authors,
            "concepts": concept_entries,
            "cited_by_count": work.get("cited_by_count"),
            "referenced_works": work.get("referenced_works") or [],
            "license": open_access.get("license"),
            "oa_status": open_access.get("oa_status"),
            "is_oa": work.get("is_oa"),
            "oa_url": (open_access.get("oa_url") or (work.get("best_oa_location") or {}).get("url")),
            "notes": None,
            "provenance": provenance,
        }

        return document

    # ------------------------------------------------------------------
    # Enrichment helpers
    # ------------------------------------------------------------------
    def _enrich_document(self, document: dict) -> dict:
        doi = document.get("doi")
        needs_abstract = not document.get("abstract") and doi
        needs_oa_url = not document.get("oa_url") and doi

        if not doi:
            return document

        if needs_abstract:
            pubmed_data = self._fetch_pubmed_metadata(doi)
            if pubmed_data.get("abstract"):
                document["abstract"] = pubmed_data["abstract"]
                self._append_provenance(document, "pubmed", pubmed_data)

        if (needs_abstract and not document.get("abstract")) or needs_oa_url:
            semscholar_data = self._fetch_semantic_scholar_metadata(doi)
            if semscholar_data.get("abstract") and not document.get("abstract"):
                document["abstract"] = semscholar_data["abstract"]
            if semscholar_data.get("oa_url") and not document.get("oa_url"):
                document["oa_url"] = semscholar_data["oa_url"]
            if semscholar_data:
                self._append_provenance(document, "semantic_scholar", semscholar_data)

        return document

    def _append_provenance(self, document: dict, source: str, payload: dict) -> None:
        provenance = document.setdefault("provenance", {})
        sources = provenance.setdefault("enrichments", [])
        sources.append({"source": source, "retrieved_at": dt.datetime.utcnow().isoformat() + "Z", "payload": payload})

    def _fetch_pubmed_metadata(self, doi: str) -> Dict[str, Optional[str]]:
        params = {
            "db": "pubmed",
            "retmode": "json",
            "term": f"{doi}[DOI]",
        }
        email = os.getenv("NCBI_EMAIL")
        if email:
            params["email"] = email
        api_key = os.getenv("PUBMED_KEY")
        if api_key:
            params["api_key"] = api_key

        try:
            response = self.session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            id_list = (data.get("esearchresult") or {}).get("idlist") or []
            if not id_list:
                return {}
            pubmed_id = id_list[0]
            summary_response = self.session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "retmode": "xml",
                    "id": pubmed_id,
                },
                timeout=self.request_timeout,
            )
            summary_response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "pubmed.enrichment.failed",
                extra={"doi": doi, "error": str(exc)},
            )
            return {}

        from xml.etree import ElementTree as ET

        try:
            root = ET.fromstring(summary_response.text)
        except ET.ParseError as exc:
            logger.warning(
                "pubmed.enrichment.parse_error",
                extra={"doi": doi, "error": str(exc)},
            )
            return {}

        abstract_texts = []
        for abstract_text in root.findall(".//Abstract/AbstractText"):
            abstract_texts.append(" ".join(abstract_text.itertext()))

        if not abstract_texts:
            return {}

        return {
            "source_id": f"pubmed:{pubmed_id}",
            "abstract": "\n".join(abstract_texts).strip(),
        }

    def _fetch_semantic_scholar_metadata(self, doi: str) -> Dict[str, Optional[str]]:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {
            "fields": "abstract,openAccessPdf",
        }
        headers = {}
        api_key = os.getenv("SEMANTIC_SCHOLAR_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.request_timeout)
            logger.info(
                "semantic_scholar.request",
                extra={"doi": doi, "status_code": response.status_code},
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning(
                "semantic_scholar.enrichment.failed",
                extra={"doi": doi, "error": str(exc)},
            )
            return {}

        return {
            "abstract": data.get("abstract"),
            "oa_url": (data.get("openAccessPdf") or {}).get("url"),
        }
