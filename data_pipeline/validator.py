"""Document & chunk validation.

Enforces payload integrity before content is embedded and indexed. Uses Pydantic
models as the schema contract and layers additional business rules on top:
non-empty text, minimum length, control-character stripping, and metadata
completeness. When ``great_expectations`` is available a lightweight expectation
suite is also evaluated; otherwise the custom checks are authoritative.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Matches C0/C1 control characters except tab, newline, and carriage return.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class DocumentSchema(BaseModel):
    """Schema contract for a chunk that is about to be embedded."""

    text: str = Field(..., min_length=1)
    source_file: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(default=0, ge=0)
    creation_date: Optional[str] = Field(default=None)


@dataclass
class ValidationResult:
    """Outcome of validating a batch of records."""

    valid_records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "valid": len(self.valid_records),
            "invalid": len(self.errors),
            "total": len(self.valid_records) + len(self.errors),
        }


class DocumentValidator:
    """Validates and sanitises document records prior to embedding."""

    def __init__(self, min_text_length: int = 3, max_text_length: int = 50_000) -> None:
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self._ge_available = self._detect_great_expectations()

    @staticmethod
    def _detect_great_expectations() -> bool:
        try:
            import great_expectations  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    # Sanitisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def strip_corrupted_characters(text: str) -> str:
        """Remove control characters and normalise unicode to NFKC form."""
        if not text:
            return ""
        normalised = unicodedata.normalize("NFKC", text)
        cleaned = _CONTROL_CHARS.sub("", normalised)
        # Collapse the excessive whitespace that stripping can leave behind.
        return re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitise a single record.

        Returns the cleaned record on success.

        Raises:
            ValueError: if the record fails schema or business-rule checks.
        """
        cleaned_text = self.strip_corrupted_characters(str(record.get("text", "")))
        if len(cleaned_text) < self.min_text_length:
            raise ValueError(
                f"Text below minimum length ({len(cleaned_text)} < {self.min_text_length})."
            )
        if len(cleaned_text) > self.max_text_length:
            cleaned_text = cleaned_text[: self.max_text_length]

        candidate = {**record, "text": cleaned_text}
        try:
            model = DocumentSchema(**candidate)
        except ValidationError as exc:
            raise ValueError(f"Schema validation failed: {exc}") from exc

        return model.model_dump() if hasattr(model, "model_dump") else model.dict()

    def validate_batch(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Validate a batch, partitioning into valid records and errors."""
        result = ValidationResult()
        for index, record in enumerate(records):
            try:
                result.valid_records.append(self.validate_record(record))
            except ValueError as exc:
                result.errors.append({"index": index, "error": str(exc), "record": record})
                logger.warning("Record %d rejected: %s", index, exc)

        if self._ge_available:
            self._run_expectation_suite(result.valid_records)

        logger.info(
            "Validation complete: %d valid, %d invalid.",
            len(result.valid_records),
            len(result.errors),
        )
        return result

    # ------------------------------------------------------------------ #
    # Optional Great Expectations integration
    # ------------------------------------------------------------------ #
    def _run_expectation_suite(self, records: List[Dict[str, Any]]) -> None:
        """Evaluate a minimal expectation suite when GE is installed."""
        try:
            import great_expectations as ge

            if not records:
                return
            dataset = ge.dataset.PandasDataset(records) if hasattr(ge, "dataset") else None
            if dataset is None:
                return
            dataset.expect_column_values_to_not_be_null("text")
            dataset.expect_column_values_to_not_be_null("source_file")
            logger.debug("Great Expectations suite evaluated on %d record(s).", len(records))
        except Exception as exc:  # GE misconfiguration must never break ingest.
            logger.debug("Skipping Great Expectations suite: %s", exc)
