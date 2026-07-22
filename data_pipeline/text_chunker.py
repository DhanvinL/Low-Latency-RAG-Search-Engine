"""Recursive character/token-aware text chunking.

Splits long documents into overlapping chunks sized for the embedding model's
context window (512 tokens with a 64-token overlap by default). The splitter is
recursive: it prefers to break on semantically meaningful separators
(paragraphs, then sentences, then words) before falling back to hard character
slicing, which keeps chunk boundaries clean.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.settings import settings

logger = logging.getLogger(__name__)

# Rough heuristic: English averages ~4 characters per token. Used to translate
# the token-based configuration into character budgets without requiring a
# tokenizer dependency at import time.
_CHARS_PER_TOKEN = 4


@dataclass
class TextChunk:
    """A single chunk of text plus inherited document metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)


class RecursiveCharacterTextChunker:
    """Recursively splits text on a descending list of separators."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size_tokens: int = None,
        chunk_overlap_tokens: int = None,
        separators: List[str] = None,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens or settings.chunk_size_tokens
        self.chunk_overlap_tokens = (
            chunk_overlap_tokens
            if chunk_overlap_tokens is not None
            else settings.chunk_overlap_tokens
        )
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be < chunk_size_tokens.")
        self.separators = separators or self.DEFAULT_SEPARATORS
        self._chunk_chars = self.chunk_size_tokens * _CHARS_PER_TOKEN
        self._overlap_chars = self.chunk_overlap_tokens * _CHARS_PER_TOKEN

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def split_text(self, text: str) -> List[str]:
        """Split raw ``text`` into a list of overlapping chunk strings."""
        cleaned = self._normalise_whitespace(text)
        if not cleaned:
            return []
        segments = self._recursive_split(cleaned, self.separators)
        return self._merge_with_overlap(segments)

    def split_document(
        self, text: str, metadata: Dict[str, Any], id_prefix: str = "chunk"
    ) -> List[TextChunk]:
        """Split ``text`` and attach ``metadata`` (plus a chunk index) to each."""
        chunks: List[TextChunk] = []
        for index, piece in enumerate(self.split_text(text)):
            chunk_meta = {**metadata, "chunk_index": index}
            chunks.append(
                TextChunk(
                    chunk_id=f"{id_prefix}-{index}",
                    text=piece,
                    metadata=chunk_meta,
                )
            )
        logger.debug("Split document into %d chunk(s).", len(chunks))
        return chunks

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse runs of spaces/tabs but preserve paragraph breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Break ``text`` into pieces no larger than the char budget."""
        if len(text) <= self._chunk_chars:
            return [text] if text else []

        if not separators:
            # Hard slice as a last resort.
            return [
                text[i : i + self._chunk_chars]
                for i in range(0, len(text), self._chunk_chars)
            ]

        separator, *rest = separators
        parts = text.split(separator) if separator else list(text)

        segments: List[str] = []
        for part in parts:
            piece = part + separator if separator else part
            if len(piece) <= self._chunk_chars:
                segments.append(piece)
            else:
                segments.extend(self._recursive_split(piece, rest))
        return [s for s in segments if s.strip()]

    def _merge_with_overlap(self, segments: List[str]) -> List[str]:
        """Greedily pack segments up to the budget, adding overlap tails."""
        chunks: List[str] = []
        current = ""
        for segment in segments:
            if len(current) + len(segment) <= self._chunk_chars:
                current += segment
                continue
            if current.strip():
                chunks.append(current.strip())
            # Seed the next chunk with the overlap tail of the previous one.
            overlap_tail = current[-self._overlap_chars :] if self._overlap_chars else ""
            current = overlap_tail + segment
            # A single oversized segment must still be hard-split.
            while len(current) > self._chunk_chars:
                chunks.append(current[: self._chunk_chars].strip())
                current = current[self._chunk_chars - self._overlap_chars :]
        if current.strip():
            chunks.append(current.strip())
        return chunks


# Backwards/alternate name referenced in the architecture spec.
SemanticTokenSplitter = RecursiveCharacterTextChunker
