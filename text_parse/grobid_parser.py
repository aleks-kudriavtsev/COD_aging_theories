"""Structured parsing of PDF/HTML assets via GROBID and related tooling.

This parser will orchestrate calls to a GROBID service, fallback to Tesseract
for OCR when PDFs lack text, and normalize the output into the document schema
expected by ontology induction. Integration hooks with the Prefect/make
orchestrator will track file provenance and text checksums.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ParsedDocument:
    """Represents a structured text artifact with metadata and content."""

    source_path: Path
    tei_xml: Optional[str] = None
    plain_text: Optional[str] = None


class GrobidParser:
    """Placeholder interface to the GROBID/Tesseract parsing toolchain."""

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Parse the provided PDF into structured text.

        The production version will invoke GROBID's processFulltextDocument API,
        capture TEI XML, generate normalized sections, and perform OCR via
        Tesseract when the PDF is image-based. Until implemented, an empty
        ``ParsedDocument`` with the source path is returned.
        """

        return ParsedDocument(source_path=pdf_path)
