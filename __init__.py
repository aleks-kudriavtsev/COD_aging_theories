"""Top-level package exposing the modular architecture for COD aging theories."""

from .ingest import OpenAlexIngestor
from .oa_resolver import UnpaywallResolver
from .text_parse import GrobidParser
from .filtering import RelevanceClassifier
from .ontology_induction import TheoryExtractor
from .graph_refine import GraphRefiner
from .query_gen import QueryGenerator
from .retrieval import SearchManager
from .eval_reporting import ReportGenerator
from .orchestrator import AgingTheoryPipeline
from .storage import DocumentStore

__all__ = [
    "OpenAlexIngestor",
    "UnpaywallResolver",
    "GrobidParser",
    "RelevanceClassifier",
    "TheoryExtractor",
    "GraphRefiner",
    "QueryGenerator",
    "SearchManager",
    "ReportGenerator",
    "AgingTheoryPipeline",
    "DocumentStore",
]
