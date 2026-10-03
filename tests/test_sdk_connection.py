"""
test_sdk_connection.py — Live probe to TypeSafe API ('jev-latest').

Stage 0 health check:
  1. Verify API authentication via models.list()
  2. Confirm 'jev-latest' is available
  3. Run a minimal Noul question (binary decision)
  4. Run a minimal Choice question (multi-class decision)
  5. Run a minimal Score question (ordinal rating)
  6. Validate response structure and field types
  7. Report latency for each primitive
"""

from __future__ import annotations

import sys
import time

import pytest

from src.client import JevClient
from src.config import MODEL_NAME

from typesafe_sdk import Choice, Noul, Score, ChoiceAnswer, NoulAnswer, ScoreAnswer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> JevClient:
    """Create a single JevClient for the entire test module."""
    c = JevClient()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Test: Authentication & Model Discovery
# ---------------------------------------------------------------------------

class TestAuthentication:
    """Verify API credentials and model availability."""

    def test_models_list(self, client: JevClient) -> None:
        """models.list() should return at least one model."""
        models = client.list_models()
        assert len(models) > 0, "No models returned — check API key."
        print(f"\n  Available models: {[m['name'] for m in models]}")

    def test_jev_latest_available(self, client: JevClient) -> None:
        """'jev-latest' must be in the available models."""
        models = client.list_models()
        model_names = [m["name"] for m in models]
        assert MODEL_NAME in model_names, (
            f"'{MODEL_NAME}' not found. Available: {model_names}"
        )
        print(f"\n  ✓ '{MODEL_NAME}' confirmed available")


# ---------------------------------------------------------------------------
# Test: Primitive Health Checks (Noul, Choice, Score)
# ---------------------------------------------------------------------------

class TestPrimitiveNoul:
    """Validate Noul (binary decision) primitive."""

    def test_noul_basic(self, client: JevClient) -> None:
        """A basic Noul question should return a float in [0, 1]."""
        result = client.ask(
            state="The weather is sunny and warm today.",
            questions={
                "is_positive": Noul(
                    instructions="Is this statement describing positive weather?"
                ),
            },
        )
        answer = result.answers["is_positive"]
        assert isinstance(answer, NoulAnswer), f"Expected NoulAnswer, got {type(answer)}"
        assert 0.0 <= answer.noul <= 1.0, f"Noul value {answer.noul} not in [0, 1]"
        print(f"\n  Noul result: P(True) = {answer.noul:.4f}")
        print(f"  Latency: {result.latency_ms:.0f}ms")
        print(f"  Model: {result.model}")

    def test_noul_sinhala_text(self, client: JevClient) -> None:
        """Noul should work with Sinhala text input."""
        sinhala_text = "අද කාලගුණය ඉතා හොඳයි."  # "The weather is very good today."
        result = client.ask(
            state=sinhala_text,
            questions={
                "is_positive": Noul(
                    instructions="Is this text expressing a positive sentiment?"
                ),
            },
        )
        answer = result.answers["is_positive"]
        assert isinstance(answer, NoulAnswer)
        assert 0.0 <= answer.noul <= 1.0
        print(f"\n  Sinhala Noul: P(True) = {answer.noul:.4f}")
        print(f"  Latency: {result.latency_ms:.0f}ms")


class TestPrimitiveChoice:
    """Validate Choice (multi-class decision) primitive."""

    def test_choice_basic(self, client: JevClient) -> None:
        """A Choice question should return a valid category and probabilities."""
        result = client.ask(
            state="This product is amazing! Best purchase I've ever made.",
            questions={
                "sentiment": Choice(
                    instructions="What is the sentiment of this text?",
                    criteria={
                        "POSITIVE": None,
                        "NEGATIVE": None,
                        "NEUTRAL": None,
                    },
                ),
            },
        )
        answer = result.answers["sentiment"]
        assert isinstance(answer, ChoiceAnswer), f"Expected ChoiceAnswer, got {type(answer)}"
        assert answer.choice in {"POSITIVE", "NEGATIVE", "NEUTRAL"}, (
            f"Choice '{answer.choice}' not in expected set"
        )
        assert 0.0 <= answer.confidence <= 1.0
        assert set(answer.probabilities.keys()) == {"POSITIVE", "NEGATIVE", "NEUTRAL"}

        prob_sum = sum(answer.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not ~1.0"

        print(f"\n  Choice result: {answer.choice} (confidence={answer.confidence:.4f})")
        print(f"  Probabilities: {answer.probabilities}")
        print(f"  Latency: {result.latency_ms:.0f}ms")

    def test_choice_sinhala_text(self, client: JevClient) -> None:
        """Choice should work with Sinhala text and Sinhala criteria."""
        sinhala_text = "මෙම නිෂ්පාදනය ඉතා නරකයි. මුදල් නාස්තියක්."
        result = client.ask(
            state=sinhala_text,
            questions={
                "sentiment": Choice(
                    instructions="Which label best describes the overall sentiment of this text?",
                    criteria={
                        "POSITIVE": None,
                        "NEGATIVE": None,
                        "NEUTRAL": None,
                        "CONFLICT": None,
                    },
                ),
            },
        )
        answer = result.answers["sentiment"]
        assert isinstance(answer, ChoiceAnswer)
        assert answer.choice in {"POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"}
        print(f"\n  Sinhala Choice: {answer.choice} (confidence={answer.confidence:.4f})")
        print(f"  Probabilities: {answer.probabilities}")
        print(f"  Latency: {result.latency_ms:.0f}ms")


class TestPrimitiveScore:
    """Validate Score (ordinal/continuous) primitive."""

    def test_score_basic(self, client: JevClient) -> None:
        """A Score question should return a score, confidence, and per-level probabilities."""
        result = client.ask(
            state="The food was acceptable but not remarkable. Average experience.",
            questions={
                "rating": Score(
                    instructions="Rate the sentiment of this text on an ordered scale.",
                    criteria=[
                        "Negative sentiment",
                        "Neutral sentiment",
                        "Positive sentiment",
                    ],
                ),
            },
        )
        answer = result.answers["rating"]
        assert isinstance(answer, ScoreAnswer), f"Expected ScoreAnswer, got {type(answer)}"
        assert isinstance(answer.score, float)
        assert 0.0 <= answer.confidence <= 1.0
        assert isinstance(answer.probabilities, dict)
        assert len(answer.probabilities) == 3

        print(f"\n  Score result: {answer.score:.4f} (confidence={answer.confidence:.4f})")
        print(f"  Probabilities: {answer.probabilities}")
        print(f"  Legend: {answer.legend}")
        print(f"  Latency: {result.latency_ms:.0f}ms")


class TestMultiQuestion:
    """Validate sending multiple questions in a single API call."""

    def test_parallel_questions(self, client: JevClient) -> None:
        """Multiple different primitives in one call should all return."""
        result = client.ask(
            state="This movie was neither good nor bad. It was just okay.",
            questions={
                "sentiment_choice": Choice(
                    instructions="What is the sentiment?",
                    criteria={"POSITIVE": None, "NEGATIVE": None, "NEUTRAL": None},
                ),
                "is_positive": Noul(
                    instructions="Is this text expressing positive sentiment?"
                ),
                "sentiment_score": Score(
                    instructions="Rate the sentiment.",
                    criteria=["Negative", "Neutral", "Positive"],
                ),
            },
        )
        assert "sentiment_choice" in result.answers
        assert "is_positive" in result.answers
        assert "sentiment_score" in result.answers
        assert isinstance(result.answers["sentiment_choice"], ChoiceAnswer)
        assert isinstance(result.answers["is_positive"], NoulAnswer)
        assert isinstance(result.answers["sentiment_score"], ScoreAnswer)

        print(f"\n  Multi-question call successful")
        print(f"  Choice: {result.answers['sentiment_choice'].choice}")
        print(f"  Noul: P(True) = {result.answers['is_positive'].noul:.4f}")
        print(f"  Score: {result.answers['sentiment_score'].score:.4f}")
        print(f"  Latency: {result.latency_ms:.0f}ms")
        print(f"  Usage: input={result.usage.input_tokens}, output={result.usage.output_tokens}")


class TestUsageMetadata:
    """Validate token usage reporting."""

    def test_usage_fields(self, client: JevClient) -> None:
        """Usage should report input_tokens and output_tokens."""
        result = client.ask(
            state="Test input.",
            questions={
                "q": Noul(instructions="Is this a test?"),
            },
        )
        usage = result.usage
        # input_tokens may be None but should typically be an int
        if usage.input_tokens is not None:
            assert isinstance(usage.input_tokens, int)
            assert usage.input_tokens > 0
            print(f"\n  Input tokens: {usage.input_tokens}")
        else:
            print("\n  Input tokens: not reported")

        if usage.output_tokens is not None:
            assert isinstance(usage.output_tokens, int)
            print(f"  Output tokens: {usage.output_tokens}")
        else:
            print("  Output tokens: not reported")
