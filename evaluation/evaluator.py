"""Automated RAG quality evaluation.

Computes the core Ragas metrics — Faithfulness, Answer Relevance, and Context
Recall — across a set of generated samples and exports the results as JSON. When
``ragas`` is installed it is used directly; otherwise a transparent,
lexical-overlap approximation of each metric is computed so the evaluation
runner always produces scores and logs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "as", "by", "at", "it", "this", "that", "be", "from",
}


@dataclass
class EvaluationSample:
    """A single (question, answer, contexts, ground_truth) evaluation record."""

    question: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None


@dataclass
class SampleScores:
    """Per-sample metric scores."""

    question: str
    faithfulness: float
    answer_relevance: float
    context_recall: float


@dataclass
class EvaluationReport:
    """Aggregate evaluation report."""

    backend: str
    num_samples: int
    faithfulness: float
    answer_relevance: float
    context_recall: float
    per_sample: List[SampleScores]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["per_sample"] = [asdict(s) for s in self.per_sample]
        return payload


class RagasEvaluator:
    """Runs Ragas (or a lexical fallback) over evaluation samples."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir or settings.evaluation_output_dir
        self._ragas_available = self._detect_ragas()

    @staticmethod
    def _detect_ragas() -> bool:
        try:
            import ragas  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self, samples: List[EvaluationSample], now: Optional[float] = None
    ) -> EvaluationReport:
        """Evaluate ``samples`` and return an aggregate report."""
        if not samples:
            raise ValueError("At least one evaluation sample is required.")

        if self._ragas_available:
            try:
                return self._evaluate_with_ragas(samples, now)
            except Exception as exc:  # network/LLM judge failure -> fallback.
                logger.warning("Ragas evaluation failed (%s); using lexical fallback.", exc)

        return self._evaluate_lexical(samples, now)

    def run_and_export(
        self,
        samples: List[EvaluationSample],
        filename: str = "evaluation_metrics.json",
        now: Optional[float] = None,
    ) -> str:
        """Evaluate and write the report to JSON. Returns the file path."""
        report = self.evaluate(samples, now=now)
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        logger.info("Evaluation report written to %s.", path)
        return path

    # ------------------------------------------------------------------ #
    # Ragas backend
    # ------------------------------------------------------------------ #
    def _evaluate_with_ragas(
        self, samples: List[EvaluationSample], now: Optional[float]
    ) -> EvaluationReport:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        scores = result if isinstance(result, dict) else dict(result)
        per_sample = [
            SampleScores(
                question=s.question,
                faithfulness=float(scores.get("faithfulness", 0.0)),
                answer_relevance=float(scores.get("answer_relevancy", 0.0)),
                context_recall=float(scores.get("context_recall", 0.0)),
            )
            for s in samples
        ]
        return self._aggregate("ragas", per_sample, now)

    # ------------------------------------------------------------------ #
    # Lexical fallback backend
    # ------------------------------------------------------------------ #
    def _evaluate_lexical(
        self, samples: List[EvaluationSample], now: Optional[float]
    ) -> EvaluationReport:
        per_sample: List[SampleScores] = []
        for sample in samples:
            context_text = " ".join(sample.contexts)
            faith = self._faithfulness(sample.answer, context_text)
            relevance = self._relevance(sample.answer, sample.question)
            recall = self._context_recall(context_text, sample.ground_truth)
            per_sample.append(
                SampleScores(
                    question=sample.question,
                    faithfulness=round(faith, 4),
                    answer_relevance=round(relevance, 4),
                    context_recall=round(recall, 4),
                )
            )
        return self._aggregate("lexical-fallback", per_sample, now)

    # -- Metric approximations -------------------------------------------- #
    @classmethod
    def _faithfulness(cls, answer: str, context: str) -> float:
        """Fraction of answer tokens supported by the retrieved context."""
        answer_tokens = cls._content_tokens(answer)
        context_tokens = cls._content_tokens(context)
        if not answer_tokens:
            return 0.0
        if not context_tokens:
            return 0.0
        supported = sum(1 for t in answer_tokens if t in context_tokens)
        return supported / len(answer_tokens)

    @classmethod
    def _relevance(cls, answer: str, question: str) -> float:
        """Token overlap between the answer and the question intent."""
        answer_tokens = cls._content_tokens(answer)
        question_tokens = cls._content_tokens(question)
        if not question_tokens or not answer_tokens:
            return 0.0
        overlap = len(question_tokens & answer_tokens)
        return overlap / len(question_tokens)

    @classmethod
    def _context_recall(cls, context: str, ground_truth: Optional[str]) -> float:
        """Fraction of ground-truth tokens present in the retrieved context."""
        if not ground_truth:
            # No reference available: fall back to context non-emptiness signal.
            return 1.0 if cls._content_tokens(context) else 0.0
        truth_tokens = cls._content_tokens(ground_truth)
        context_tokens = cls._content_tokens(context)
        if not truth_tokens:
            return 0.0
        recalled = sum(1 for t in truth_tokens if t in context_tokens)
        return recalled / len(truth_tokens)

    @staticmethod
    def _content_tokens(text: str) -> set:
        return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def _aggregate(
        self, backend: str, per_sample: List[SampleScores], now: Optional[float]
    ) -> EvaluationReport:
        n = len(per_sample)
        mean = lambda key: round(sum(getattr(s, key) for s in per_sample) / n, 4) if n else 0.0
        report = EvaluationReport(
            backend=backend,
            num_samples=n,
            faithfulness=mean("faithfulness"),
            answer_relevance=mean("answer_relevance"),
            context_recall=mean("context_recall"),
            per_sample=per_sample,
            timestamp=now if now is not None else time.time(),
        )
        logger.info(
            "Evaluation [%s]: faithfulness=%.3f relevance=%.3f recall=%.3f (n=%d).",
            backend,
            report.faithfulness,
            report.answer_relevance,
            report.context_recall,
            n,
        )
        return report


__all__ = ["RagasEvaluator", "EvaluationSample", "EvaluationReport", "SampleScores"]
