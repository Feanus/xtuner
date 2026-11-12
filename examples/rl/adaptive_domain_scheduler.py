"""Scheduler fragment for an existing, fully configured XTuner RL recipe.

The priors below are illustrative inputs, not measured benchmark settings. Supply
real TaskSpecs with these three task_name identities and equal prompt_repeat_k.
This module does not create engines, sandboxes, resources, or training jobs.
"""

from xtuner.v1.rl.agent_loop_manager import (
    AdaptiveDomainSchedulerConfig,
    DisaggAgentLoopManagerConfig,
    DisaggTaskSpecConfig,
)


scheduler_config = AdaptiveDomainSchedulerConfig(
    target_groups={"cyber": 12, "swe": 10, "tb": 10},  # 32 prompt groups per learner handoff
    initial_acceptance_rates={"cyber": 1.0, "swe": 1.0, "tb": 1.0},
    selection_alpha=0.5,
    acceptance_floor=0.01,
    acceptance_ewma_alpha=0.1,
    concurrency_pbar=0.75,
    concurrency_pmin=0.0,
    concurrency_pmax=1.5,
    initial_duration_seconds={"cyber": 600, "swe": 300, "tb": 1200},
    duration_ewma_alpha=0.1,
    max_concurrent_samples=512,  # With K=16: at most 32 whole pending groups in total.
    submission_granularity="group",  # Original-sibling sample backfill is not implemented.
    group_start_interval=3.0,
    buffer_capacity_factor=1.0,  # Shared completed FIFO holds 32 groups.
    min_trainable_siblings=12,
    unused_completed_groups="drop",  # Accepted surplus is discarded when pausing a full FIFO.
    seed=42,
)


def make_manager_config(tasks: list[DisaggTaskSpecConfig]) -> DisaggAgentLoopManagerConfig:
    """Use with RLDisaggregatedTrainerConfig, retaining each task's native adapter."""
    return DisaggAgentLoopManagerConfig(tasks=tasks, domain_scheduler=scheduler_config)
