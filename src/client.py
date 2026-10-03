"""
TypeSafe SDK Wrapper with retry, latency tracking, and telemetry.

Provides a high-level interface around ``typesafe_sdk.TypeSafeClient``
with:
  * Automatic retry via the SDK's built-in ``RetryPolicy``
  * Per-call wall-clock latency measurement
  * Structured result objects that carry latency + usage metadata
  * ``models.list()`` health-check helper
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
    Usage,
)

from src.config import ExperimentConfig, get_api_key


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

@dataclass
class JevCallResult:
    """Wraps a single ``system_one`` call with timing metadata."""

    response: SystemOneResponse
    latency_ms: float
    error: str | None = None

    @property
    def answers(self) -> dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]:
        return self.response.answers

    @property
    def usage(self) -> Usage:
        return self.response.usage

    @property
    def model(self) -> str:
        return self.response.model


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------

class JevClient:
    """High-level wrapper around TypeSafe SDK for the Jev probe.

    Parameters
    ----------
    config : ExperimentConfig, optional
        Experiment configuration. Loaded from YAML if not provided.
    api_key : str, optional
        Override the API key (otherwise loaded from env).
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or ExperimentConfig.from_yaml()
        self._api_key = api_key or get_api_key()

        self._retry_policy = RetryPolicy(
            max_retries=self.config.max_retries,
            backoff_initial=self.config.backoff_initial,
            backoff_max=self.config.backoff_max,
            backoff_jitter=self.config.backoff_jitter,
        )

        self._client = TypeSafeClient(
            api_key=self._api_key,
            model=self.config.model,
            retry=self._retry_policy,
            timeout=60.0,
        )

    # -- public helpers -------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """List available models. Used as an authentication health-check."""
        resp = self._client.models.list()
        # resp is a ListModelsResponse with tuple of ModelMetadata
        # ModelMetadata has: name, description, release_date
        return [
            {
                "name": m.name,
                "description": m.description,
                "release_date": m.release_date,
            }
            for m in resp.models
        ]

    def ask(
        self,
        state: str,
        questions: dict[str, Noul | Choice | Score],
        *,
        model: str | None = None,
    ) -> JevCallResult:
        """Send a ``system_one`` call and return a timed result.

        Parameters
        ----------
        state : str
            The input text (Sinhala content) presented to the model.
        questions : dict
            Named questions mapping question-key → Noul/Choice/Score.
        model : str, optional
            Override model for this call (defaults to config).

        Returns
        -------
        JevCallResult
            Includes the raw SDK response plus wall-clock latency.
        """
        t0 = time.perf_counter()
        try:
            response = self._client.system_one(
                state=state,
                questions=questions,
                model=model,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return JevCallResult(response=response, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            raise RuntimeError(
                f"TypeSafe API call failed after {latency_ms:.0f}ms: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
