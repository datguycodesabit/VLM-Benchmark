"""Shared retry, quota, and spend-estimation primitives."""

from __future__ import annotations

import math
import time
from typing import Any


class RetryableProviderError(RuntimeError):
    """Provider error classified safe for a bounded managed retry."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        category: str = "transient_provider_error",
    ) -> None:
        super().__init__(message)
        if retry_after_seconds is not None and (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, (int, float))
            or not math.isfinite(retry_after_seconds)
            or retry_after_seconds < 0
        ):
            raise ValueError("retry_after_seconds must be a finite non-negative number")
        self.retry_after_seconds = retry_after_seconds
        self.category = category


class SpendLedger:
    """Invocation-wide estimated spend across all selected models."""

    def __init__(self, limit: float | None) -> None:
        self.limit = limit
        self.estimated_usd = 0.0
        self.reserved_usd = 0.0
        self.unknown_spend_count = 0

    def can_schedule(self, reserve_usd: float = 0.0) -> bool:
        if self.limit is None:
            return True
        if self.unknown_spend_count:
            return False
        total = self.estimated_usd + self.reserved_usd
        return total < self.limit and total + reserve_usd <= self.limit

    def finish(self, estimate_usd: float | None, reserved_usd: float = 0.0) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - reserved_usd)
        if estimate_usd is None:
            self.unknown_spend_count += 1
        else:
            self.estimated_usd += estimate_usd


class CostMeter:
    """Price lookup for one model using an invocation-wide ledger."""

    def __init__(
        self,
        selector: str,
        costs: dict[str, Any],
        ledger: SpendLedger,
        *,
        provider: str,
    ) -> None:
        self.selector = selector
        self.ledger = ledger
        self.provider = provider
        api = costs.get("api", {}) if isinstance(costs, dict) else {}
        local = costs.get("local", {}) if isinstance(costs, dict) else {}
        self.assumptions = api if isinstance(api, dict) else {}
        if self.assumptions.get("model") not in {None, selector}:
            self.assumptions = {}
        self.local_assumptions = local if isinstance(local, dict) else {}
        if self.local_assumptions.get("model") not in {None, selector}:
            self.local_assumptions = {}

    @property
    def fixed_per_request(self) -> float | None:
        value = self.assumptions.get("cost_per_sample_usd")
        if value is not None:
            return float(value)
        if self.provider not in {"openai", "chatgpt"}:
            value = self.local_assumptions.get("operating_cost_per_sample_usd")
            return float(value) if value is not None else None
        return None

    @property
    def applicable(self) -> bool:
        if self.fixed_per_request is not None:
            return True
        if self.provider not in {"openai", "chatgpt"}:
            return False
        return (
            "input_per_million_usd" in self.assumptions
            and "output_per_million_usd" in self.assumptions
        )

    def can_schedule(self) -> bool:
        return self.ledger.can_schedule(self.fixed_per_request or 0.0)

    def reserve(self) -> float:
        amount = self.fixed_per_request or 0.0
        self.ledger.reserved_usd += amount
        return amount

    def finish(
        self,
        response: dict[str, Any] | None = None,
        *,
        retryable_rejection: bool = False,
        reserved_usd: float = 0.0,
    ) -> float | None:
        if retryable_rejection:
            estimate = 0.0
        else:
            estimate = self.estimate(response)
        self.ledger.finish(estimate, reserved_usd)
        return estimate

    def estimate(self, response: dict[str, Any] | None = None) -> float | None:
        fixed = self.fixed_per_request
        if fixed is not None:
            return fixed
        if self.provider not in {"openai", "chatgpt"}:
            return 0.0
        if not isinstance(response, dict):
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            details = response.get("provider_details")
            usage = details.get("usage") if isinstance(details, dict) else None
        if not isinstance(usage, dict):
            return None
        if (
            "input_per_million_usd" not in self.assumptions
            or "output_per_million_usd" not in self.assumptions
        ):
            return None
        input_tokens = _usage(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage(usage, "output_tokens", "completion_tokens")
        cached_tokens = _usage(usage, "cached_input_tokens", "cached_tokens")
        input_details = usage.get("input_tokens_details")
        if cached_tokens is None and isinstance(input_details, dict):
            cached_tokens = _usage(input_details, "cached_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        cached_tokens = cached_tokens or 0
        uncached_tokens = max(0, input_tokens - cached_tokens)
        input_rate = float(self.assumptions["input_per_million_usd"])
        cached_rate = float(self.assumptions.get("cached_input_per_million_usd", input_rate))
        output_rate = float(self.assumptions["output_per_million_usd"])
        return (
            uncached_tokens * input_rate + cached_tokens * cached_rate + output_tokens * output_rate
        ) / 1_000_000


def _usage(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def validate_execution_controls(
    *,
    max_retries: int,
    concurrency: int,
    max_requests: int | None,
    max_spend_usd: float | None,
) -> None:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if max_requests is not None and (
        isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests <= 0
    ):
        raise ValueError("max_requests must be a positive integer")
    if max_spend_usd is not None and (
        isinstance(max_spend_usd, bool)
        or not isinstance(max_spend_usd, (int, float))
        or not math.isfinite(max_spend_usd)
        or max_spend_usd <= 0
    ):
        raise ValueError("max_spend_usd must be a positive finite number")


def retry_delay(error: RetryableProviderError, retry_number: int) -> float:
    if error.retry_after_seconds is not None:
        return float(error.retry_after_seconds)
    return min(0.1 * retry_number, 1.0)


def wait_retry(error: RetryableProviderError, retry_number: int) -> float:
    delay = retry_delay(error, retry_number)
    if delay:
        time.sleep(delay)
    return delay
