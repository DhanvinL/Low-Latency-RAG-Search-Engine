"""vLLM inference client wrapper.

Exposes a uniform :class:`VLLMClient` that talks to an OpenAI-compatible vLLM
server. When the ``openai`` SDK or a live endpoint is unavailable, a mock
generator returns a deterministic, context-grounded response so the RAG chain
remains end-to-end runnable. Also documents the recommended ``AsyncLLMEngine``
server bootstrap for a real deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    """Generation sampling parameters (mirrors vLLM's SamplingParams)."""

    temperature: float = settings.llm_temperature
    max_tokens: int = settings.llm_max_tokens
    top_p: float = settings.llm_top_p
    stop: Optional[List[str]] = None

    def to_openai_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        return kwargs


@dataclass
class GenerationResult:
    """Result of an LLM generation call."""

    text: str
    model: str
    backend: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class VLLMClient:
    """OpenAI-compatible client for a vLLM inference server."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        default_params: Optional[SamplingParams] = None,
    ) -> None:
        self.endpoint = endpoint or settings.vllm_endpoint
        self.model = model or settings.vllm_model
        self.api_key = api_key or settings.vllm_api_key
        self.timeout = timeout or settings.llm_request_timeout
        self.default_params = default_params or SamplingParams()
        self._client = self._init_client()

    def _init_client(self) -> Any:
        """Construct an OpenAI client pointed at the vLLM endpoint."""
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.endpoint,
                api_key=self.api_key or "EMPTY",
                timeout=self.timeout,
            )
            logger.info("Initialised vLLM client at %s (model=%s).", self.endpoint, self.model)
            return client
        except Exception as exc:  # SDK missing -> mock generation.
            logger.warning("OpenAI SDK/vLLM endpoint unavailable (%s); using mock.", exc)
            return None

    @property
    def backend(self) -> str:
        return "vllm-openai" if self._client is not None else "mock"

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[SamplingParams] = None,
    ) -> GenerationResult:
        """Generate a completion for ``prompt`` using chat completions."""
        params = params or self.default_params
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self._client is None:
            return self._mock_generate(prompt, system_prompt, params)

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params.to_openai_kwargs(),
            )
            choice = completion.choices[0]
            usage = getattr(completion, "usage", None)
            return GenerationResult(
                text=(choice.message.content or "").strip(),
                model=self.model,
                backend=self.backend,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
        except Exception as exc:
            logger.error("vLLM generation failed (%s); returning mock response.", exc)
            return self._mock_generate(prompt, system_prompt, params)

    # ------------------------------------------------------------------ #
    # Mock generation
    # ------------------------------------------------------------------ #
    def _mock_generate(
        self, prompt: str, system_prompt: Optional[str], params: SamplingParams
    ) -> GenerationResult:
        """Deterministic, context-aware fallback generation.

        Extracts the retrieved context block from the prompt (if present) and
        composes a concise grounded answer so downstream evaluation still has
        meaningful text to score.
        """
        context_excerpt = self._extract_context(prompt)
        question = self._extract_question(prompt)
        if context_excerpt:
            snippet = context_excerpt[: params.max_tokens * 4]
            answer = (
                f"Based on the retrieved context, here is the answer to "
                f'"{question}": {snippet.strip()[:400]}'
            )
        else:
            answer = (
                f'I do not have sufficient retrieved context to answer "{question}" '
                "with confidence."
            )
        return GenerationResult(
            text=answer,
            model=f"{self.model} (mock)",
            backend="mock",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
        )

    @staticmethod
    def _extract_context(prompt: str) -> str:
        marker = "Context:"
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            return tail.split("Question:", 1)[0].strip()
        return ""

    @staticmethod
    def _extract_question(prompt: str) -> str:
        marker = "Question:"
        if marker in prompt:
            return prompt.split(marker, 1)[1].strip().splitlines()[0].strip()
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""


# --------------------------------------------------------------------------- #
# Reference: real vLLM server bootstrap (documented, not executed on import).
# --------------------------------------------------------------------------- #
def build_async_engine(model: Optional[str] = None) -> Any:  # pragma: no cover
    """Construct a vLLM ``AsyncLLMEngine`` for an in-process server.

    This is provided as a reference for GPU deployments. It is intentionally
    lazy and guarded so importing this module never requires vLLM to be present.
    """
    from vllm import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model or settings.vllm_model,
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


_client: Optional[VLLMClient] = None


def get_vllm_client() -> VLLMClient:
    """Return a cached, process-wide :class:`VLLMClient`."""
    global _client
    if _client is None:
        _client = VLLMClient()
    return _client
