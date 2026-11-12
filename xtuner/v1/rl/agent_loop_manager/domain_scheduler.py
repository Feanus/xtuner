"""Opt-in, group-based adaptive domain admission.

Task names are scheduling identities. Dataset/native ``data_source`` fields are
never used for routing or rewritten by this module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class FixedDomainSchedulerConfig(BaseModel):
    """The existing fixed weighted per-task quotas (default)."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["fixed"] = "fixed"


class AdaptiveDomainSchedulerConfig(BaseModel):
    """Soft batch targets, filter acceptance correction and optional domain caps.

    ``max_concurrent_samples`` is an independent aggregate limit, rounded down
    to a whole group. Only whole-group release is supported; original-sibling
    completion events across Ray workers are required for sample backfill.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["adaptive"] = "adaptive"
    target_groups: dict[str, PositiveInt]
    initial_acceptance_rates: dict[str, Probability]
    selection_alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    acceptance_floor: float = Field(default=0.01, gt=0, le=1, allow_inf_nan=False)
    acceptance_ewma_alpha: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)
    concurrency_pbar: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    concurrency_pmin: float = Field(default=0, ge=0, allow_inf_nan=False)
    concurrency_pmax: float = Field(default=1, ge=0, allow_inf_nan=False)
    initial_duration_seconds: dict[str, PositiveSeconds] | None = None
    duration_ewma_alpha: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)
    max_concurrent_samples: PositiveInt | None = None
    submission_granularity: Literal["group"] = "group"
    group_start_interval: float = Field(default=0, ge=0, allow_inf_nan=False)
    buffer_capacity_factor: float = Field(default=2, ge=1, allow_inf_nan=False)
    min_trainable_siblings: int = Field(default=0, ge=0, strict=True)
    unused_completed_groups: Literal["drop"] = "drop"
    seed: int = 42

    @model_validator(mode="after")
    def validate_domains(self):
        if not self.target_groups or any(not name.strip() for name in self.target_groups):
            raise ValueError("Adaptive targets require nonempty task names.")
        if self.initial_acceptance_rates.keys() != self.target_groups.keys():
            raise ValueError("initial_acceptance_rates must cover exactly target_groups.")
        # Rejections can eventually drive any prior to zero. Validate the
        # denominator floor's worst case before starting any worker.
        required = {name: b / self.acceptance_floor for name, b in self.target_groups.items()}
        if not math.isfinite(self.batch_size / self.acceptance_floor) or not all(
            math.isfinite(value) for value in required.values()
        ):
            raise ValueError("target_groups / effective acceptance must be finite.")
        if self.concurrency_pbar is not None and not all(
            math.isfinite((1 + self.concurrency_pmax) * value) for value in required.values()
        ):
            raise ValueError("Adaptive concurrency budgets must remain finite at the acceptance floor.")
        if not math.isfinite(self.buffer_capacity_factor * self.batch_size):
            raise ValueError("Completed FIFO capacity must be finite.")
        if self.concurrency_pbar is None:
            if self.initial_duration_seconds is not None:
                raise ValueError("initial_duration_seconds requires concurrency_pbar.")
        else:
            if not self.concurrency_pmin <= self.concurrency_pbar <= self.concurrency_pmax:
                raise ValueError("Expected 0 <= concurrency_pmin <= concurrency_pbar <= concurrency_pmax.")
            if (
                self.initial_duration_seconds is None
                or self.initial_duration_seconds.keys() != self.target_groups.keys()
            ):
                raise ValueError("initial_duration_seconds must cover exactly target_groups with adaptive caps.")
            solve_domain_budgets(
                {
                    name: b / max(self.initial_acceptance_rates[name], self.acceptance_floor)
                    for name, b in self.target_groups.items()
                },
                self.initial_duration_seconds,
                p_min=self.concurrency_pmin,
                p_max=self.concurrency_pmax,
                p_bar=self.concurrency_pbar,
            )
        return self

    @property
    def batch_size(self) -> int:
        return sum(self.target_groups.values())

    @property
    def buffer_capacity(self) -> int:
        return math.floor(self.buffer_capacity_factor * self.batch_size)


@dataclass(frozen=True)
class DomainBudgetSolution:
    c: float
    oversampling: dict[str, float]
    float_caps: dict[str, float]
    group_caps: dict[str, int]


def solve_domain_budgets(
    required_groups: dict[str, float],
    durations_seconds: dict[str, float],
    *,
    p_min: float,
    p_max: float,
    p_bar: float,
) -> DomainBudgetSolution:
    """Solve the clipped m-weighted oversampling mean; c has units 1/s."""
    if not required_groups or required_groups.keys() != durations_seconds.keys():
        raise ValueError("Budget domains must be nonempty and match duration domains.")
    if not all(math.isfinite(m) and m >= 1 for m in required_groups.values()):
        raise ValueError("Required groups must be finite and >= 1.")
    if not all(math.isfinite(t) and t > 0 for t in durations_seconds.values()):
        raise ValueError("Durations must be finite and positive.")
    if not all(math.isfinite(p) for p in (p_min, p_bar, p_max)) or not 0 <= p_min <= p_bar <= p_max:
        raise ValueError("Expected finite 0 <= p_min <= p_bar <= p_max.")
    scale = max(required_groups.values())
    weights = {name: m / scale for name, m in required_groups.items()}
    target = p_bar * math.fsum(weights.values())
    lo = 0.0
    hi = (1 + p_max) / min(durations_seconds.values())
    if not math.isfinite(hi):
        raise ValueError("Duration estimates are too small for finite budget scaling.")

    def proportions(c):
        return {name: min(p_max, max(p_min, c * t - 1)) for name, t in durations_seconds.items()}

    if p_bar == p_min:
        c = lo
    elif p_bar == p_max:
        c = hi
    else:
        for _ in range(64):
            c = lo + (hi - lo) / 2
            total = math.fsum(weights[name] * p for name, p in proportions(c).items())
            if total < target:
                lo = c
            else:
                hi = c
        c = lo + (hi - lo) / 2
        free = [name for name, t in durations_seconds.items() if p_min < c * t - 1 < p_max]
        if free:
            fixed = math.fsum(weights[name] * p for name, p in proportions(c).items() if name not in free)
            refined = (target - fixed + math.fsum(weights[name] for name in free)) / math.fsum(
                weights[name] * durations_seconds[name] for name in free
            )
            if all(p_min <= refined * durations_seconds[name] - 1 <= p_max for name in free):
                c = refined
    p = proportions(c)
    caps = {name: (1 + p[name]) * m for name, m in required_groups.items()}
    if not all(math.isfinite(cap) for cap in caps.values()):
        raise ValueError("Domain budgets must be finite.")
    return DomainBudgetSolution(c, p, caps, {name: math.floor(cap) for name, cap in caps.items()})


class DomainSchedulingState:
    """Online estimates and seeded draws, owned by one manager event loop."""

    def __init__(self, config: AdaptiveDomainSchedulerConfig):
        self.config = config
        self.acceptance = dict(config.initial_acceptance_rates)
        self.acceptance_observations = dict.fromkeys(config.target_groups, 0)
        self.durations = dict(config.initial_duration_seconds or {})
        self.duration_observations = dict.fromkeys(config.target_groups, 0)
        self.in_flight = dict.fromkeys(config.target_groups, 0)
        self.rng = random.Random(config.seed)
        self.solution: DomainBudgetSolution | None = None
        self.window_counts = dict.fromkeys(config.target_groups, 0)
        self.selection_weights = dict.fromkeys(config.target_groups, 0.0)
        self.refresh_caps()

    def required_groups(self) -> dict[str, float]:
        return {
            name: b / max(self.acceptance[name], self.config.acceptance_floor)
            for name, b in self.config.target_groups.items()
        }

    def refresh_caps(self) -> None:
        if self.config.concurrency_pbar is not None:
            self.solution = solve_domain_budgets(
                self.required_groups(),
                self.durations,
                p_min=self.config.concurrency_pmin,
                p_max=self.config.concurrency_pmax,
                p_bar=self.config.concurrency_pbar,
            )

    def observe(self, domain: str, *, duration_seconds: float, filter_verdict: bool | None) -> None:
        if domain in self.durations:
            a = self.config.duration_ewma_alpha
            self.durations[domain] = (1 - a) * self.durations[domain] + a * duration_seconds
        self.duration_observations[domain] += 1
        if filter_verdict is not None:
            a = self.config.acceptance_ewma_alpha
            self.acceptance[domain] = (1 - a) * self.acceptance[domain] + a * int(filter_verdict)
            self.acceptance_observations[domain] += 1
        self.refresh_caps()

    def draw(self, counts: dict[str, int], eligible: list[str]) -> str | None:
        self.window_counts = dict(counts)
        a = self.config.selection_alpha
        self.selection_weights = {
            name: (a * b + (1 - a) * max(b - counts.get(name, 0), 0))
            / max(self.acceptance[name], self.config.acceptance_floor)
            for name, b in self.config.target_groups.items()
        }
        if not eligible:
            return None
        weights = [self.selection_weights[name] for name in eligible]
        if not any(weights):
            required = self.required_groups()
            weights = [required[name] for name in eligible]
        return self.rng.choices(eligible, weights=weights, k=1)[0]

    def metrics(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, b in self.config.target_groups.items():
            prefix = f"scheduler/{name}/"
            values.update(
                {
                    prefix + "target_groups": b,
                    prefix + "batch_window_groups": self.window_counts[name],
                    prefix + "selection_weight": self.selection_weights[name],
                    prefix + "acceptance_ewma": self.acceptance[name],
                    prefix + "acceptance_observations": self.acceptance_observations[name],
                    prefix + "duration_observations": self.duration_observations[name],
                    prefix + "in_flight_groups": self.in_flight[name],
                }
            )
            if self.solution is not None:
                values.update(
                    {
                        prefix + "duration_seconds": self.durations[name],
                        prefix + "required_groups": self.required_groups()[name],
                        prefix + "oversampling": self.solution.oversampling[name],
                        prefix + "float_group_cap": self.solution.float_caps[name],
                        prefix + "group_cap": self.solution.group_caps[name],
                    }
                )
        if self.solution is not None:
            values["scheduler/c_per_second"] = self.solution.c
        return values
