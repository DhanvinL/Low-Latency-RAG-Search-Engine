"""Document ingestion utilities.

Parses PDF (and plain-text) source documents into structured
:class:`LoadedDocument` records, capturing per-page text alongside provenance
metadata (source file, page number, creation date). The loader degrades
gracefully: if neither ``pdfplumber`` nor ``PyPDF2`` is installed it falls back
to reading the file as UTF-8 text so the pipeline remains exercisable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadedPage:
    """A single extracted page of text with page-scoped metadata."""

    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """A parsed document composed of one or more pages."""

    source_file: str
    pages: List[LoadedPage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all page text with form-feed separators."""
        return "\n\f\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PDFLoader:
    """Loads documents from disk and extracts text plus metadata."""

    SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")

    def __init__(self) -> None:
        self._backend = self._detect_backend()
        logger.info("PDFLoader initialised with backend='%s'.", self._backend)

    @staticmethod
    def _detect_backend() -> str:
        """Choose the best available PDF parsing backend."""
        try:
            import pdfplumber  # noqa: F401

            return "pdfplumber"
        except ImportError:
            pass
        try:
            import PyPDF2  # noqa: F401

            return "pypdf2"
        except ImportError:
            pass
        return "plaintext"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self, path: str) -> LoadedDocument:
        """Load a single document from ``path``.

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if the file extension is unsupported.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}' for path {path}.")

        base_metadata = self._file_metadata(path)

        if ext == ".pdf" and self._backend != "plaintext":
            pages = self._load_pdf(path, base_metadata)
        else:
            pages = self._load_text(path, base_metadata)

        logger.info("Loaded '%s' (%d page(s)).", path, len(pages))
        return LoadedDocument(source_file=path, pages=pages, metadata=base_metadata)

    def load_many(self, paths: List[str]) -> List[LoadedDocument]:
        """Load multiple documents, skipping (and logging) failures."""
        documents: List[LoadedDocument] = []
        for path in paths:
            try:
                documents.append(self.load(path))
            except Exception as exc:  # keep batch ingestion resilient.
                logger.error("Failed to load '%s': %s", path, exc)
        return documents

    # ------------------------------------------------------------------ #
    # Backend implementations
    # ------------------------------------------------------------------ #
    def _load_pdf(self, path: str, base_metadata: Dict[str, Any]) -> List[LoadedPage]:
        if self._backend == "pdfplumber":
            return self._load_pdf_pdfplumber(path, base_metadata)
        return self._load_pdf_pypdf2(path, base_metadata)

    def _load_pdf_pdfplumber(
        self, path: str, base_metadata: Dict[str, Any]
    ) -> List[LoadedPage]:
        import pdfplumber

        pages: List[LoadedPage] = []
        with pdfplumber.open(path) as pdf:
            doc_created = self._pdf_creation_date(getattr(pdf, "metadata", None))
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(
                    LoadedPage(
                        page_number=index,
                        text=text,
                        metadata={
                            **base_metadata,
                            "page_number": index,
                            "creation_date": doc_created,
                        },
                    )
                )
        return pages

    def _load_pdf_pypdf2(
        self, path: str, base_metadata: Dict[str, Any]
    ) -> List[LoadedPage]:
        import PyPDF2

        pages: List[LoadedPage] = []
        with open(path, "rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            doc_info = getattr(reader, "metadata", None)
            doc_created = self._pdf_creation_date(
                {"CreationDate": getattr(doc_info, "creation_date", None)}
                if doc_info
                else None
            )
            for index, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append(
                    LoadedPage(
                        page_number=index,
                        text=text,
                        metadata={
                            **base_metadata,
                            "page_number": index,
                            "creation_date": doc_created,
                        },
                    )
                )
        return pages

    def _load_text(self, path: str, base_metadata: Dict[str, Any]) -> List[LoadedPage]:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        return [
            LoadedPage(
                page_number=1,
                text=text,
                metadata={
                    **base_metadata,
                    "page_number": 1,
                    "creation_date": base_metadata.get("creation_date"),
                },
            )
        ]

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _file_metadata(path: str) -> Dict[str, Any]:
        stat = os.stat(path)
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
        return {
            "source_file": os.path.basename(path),
            "source_path": os.path.abspath(path),
            "creation_date": created,
            "file_size_bytes": stat.st_size,
        }

    @staticmethod
    def _pdf_creation_date(pdf_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Normalise a PDF ``CreationDate`` field into an ISO-8601 string."""
        if not pdf_metadata:
            return None
        raw = pdf_metadata.get("CreationDate") or pdf_metadata.get("creation_date")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw.astimezone(timezone.utc).isoformat()
        # PDF dates often look like ``D:20240131120000Z``.
        text = str(raw).lstrip("D:").split("+")[0].split("Z")[0]
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(text[: len(fmt.replace("%", "") + "00")], fmt).isoformat()
            except (ValueError, IndexError):
                continue
        return str(raw)
