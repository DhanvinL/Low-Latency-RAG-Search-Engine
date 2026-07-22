"""Evaluation package: Ragas-based automated quality benchmarking."""

from evaluation.evaluator import (
    EvaluationReport,
    EvaluationSample,
    RagasEvaluator,
    SampleScores,
)

__all__ = ["RagasEvaluator", "EvaluationSample", "EvaluationReport", "SampleScores"]
