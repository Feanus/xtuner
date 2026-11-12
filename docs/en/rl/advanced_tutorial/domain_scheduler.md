# Adaptive domain scheduling

XTuner keeps fixed weighted per-task batch quotas by default. Set
`domain_scheduler=AdaptiveDomainSchedulerConfig(...)` on a generic
`DisaggAgentLoopManagerConfig` to opt into soft domain targets, inverse filter
acceptance weighting, optional duration-based concurrency caps, and one globally
paced admission stream. `FixedDomainSchedulerConfig()` explicitly selects the
existing default. Task weights and the fixed production path are unchanged.

`AgentLoopManagerConfig` accepts the same choice for a colocated run. Its adaptive
producer stops and drains at each batch boundary before training. Continuous
generation overlapping learner training uses the disaggregated manager and the
existing `RLDisaggregatedTrainerConfig` lifecycle.

## Identity and units

A TaskSpec's unique `task_name` is its scheduling domain, replay routing identity,
and sampler/checkpoint identity. Use distinct TaskSpecs for Cyber, SWE and TB even
if they share an agent-loop implementation. Native `data_source`, adapter routing,
and dataset metadata remain unchanged. Different domains retain independent
sampler cursors and the dataloader's existing shuffle/epoch behavior.

A group is one prompt with `SamplerConfig.prompt_repeat_k` original siblings.
Adaptive scheduling currently requires the same positive sibling count `K` for
all domains. `target_groups` contains positive integers `B_i` for exactly the
manager's task names. Their sum must equal the `batch_size` supplied by the
trainer. Batching and caps count groups, including sandbox groups whose original
siblings expand into multiple training segments. A configured
`min_trainable_siblings` counts surviving original sibling identities, never
expanded segments.

Targets are soft. A 12/20 target may produce a batch with another composition.
`TaskSpec.weight` is used by fixed scheduling; adaptive scheduling uses its
explicit `target_groups`, without modifying weights or requiring per-task quotas.

## Admission and collection

For each admission, eligible domains are drawn using a seeded weighted random
choice:

```text
r_eff_i = max(r_i, acceptance_floor)
w_i = [selection_alpha * B_i
       + (1 - selection_alpha) * max(B_i - A_i, 0)] / r_eff_i
P(i | eligible) = w_i / sum(w_j for eligible domains j)
```

If every eligible weight is zero, admission falls back to `B_i / r_eff_i`.
The seed controls the domain RNG; completion timing can change future draws in
an asynchronous run.

`A_i` counts the next batch window: selected partial groups plus the leading
completed FIFO groups that fit the remaining batch slots. This implementation
reserves no partial batch: the consumer atomically takes all `G` groups under the
replay lock or leaves the buffer untouched. Consequently its window is exactly
the raw next `G` completed FIFO records. In-flight groups and previous batches
never contribute. Queued stale groups contribute until native staleness refresh
rejects them. FIFO snapshots/takes share the replay lock; the learner receives
global completion order, without per-task reordering or staleness priority.

The completed FIFO holds `floor(buffer_capacity_factor * G)` groups. When full,
new admissions stop. Already running groups may complete into a separate bounded
waiting wave; its size is bounded by the aggregate pending-group limit at the
last admission. Timings, verdicts, infrastructure failures and slot releases
continue to be collected while pacing or backpressure stops starts. No new
wave starts until those results fit in the FIFO.

`group_start_interval` is minimum spacing between whole-group submissions across
all domains. The first starts immediately. Pauses and backpressure accumulate no
catch-up credits. Runtime semaphores, inference limits and sandbox capacity remain
independent downstream limits.

## Acceptance and duration estimates

Acceptance is **RL dynamic-filter pass probability**, not task success, reward
mean or infrastructure availability. Only an explicit verdict from
`TaskSpec.is_valid_sample_fn` updates it:

```text
r_i <- (1 - acceptance_ewma_alpha) * r_i
       + acceptance_ewma_alpha * int(filter_passed)
```

The group must first complete, meet the original-sibling admission threshold,
and supply finite `reward['score']` values for surviving training entries.
Each entry must satisfy native response/input/label length checks and have at
least one loss-bearing response token (or shifted agentic label). A malformed or
fully masked training entry rejects the group before filtering, as the native
trainer could not consume that group. Minimum counts use usable original IDs.
Admission failures and missing rewards do not invoke the filter and supply no
acceptance observation. Passes and filter rejects each supply one observation
per generated original group. With filtering disabled, priors stay unchanged.
Later weight/token staleness does not create another filter trial. Constant
positive rewards can fail a variance filter just as constant zero rewards can.

Built-in base and sandbox loops preserve this provenance through
`maybe_filter_invalid_sample`. Custom `generate_group` overrides must preserve
`extra_fields['_domain_scheduler']` on each original sibling/derived segment and
call that helper after generation and reward processing, before deferred Teacher
scoring. The scheduler never guesses a filter verdict from output status or
reward. Existing Ray actors and routers carry the provenance with rollout states;
no process-local callback is used to infer sibling completion.

Duration starts when the whole generation coroutine launches and ends when
siblings, reward work and group processing return. Native semaphore/runtime waits
and pauses experienced by that attempt are included. FIFO insertion waits and
learner training are excluded. Every normally returned attempt, including an
infrastructure/admission failure, missing-reward group or filter reject, supplies
a duration observation. Raised exceptions and cancelled attempts supply none.

```text
t_i <- (1 - duration_ewma_alpha) * t_i
       + duration_ewma_alpha * completed_group_seconds
```

The three alpha values are independent. Priors must be finite and cover exactly
the domains; duration priors must be positive when adaptive caps are enabled.

## Adaptive concurrency and the aggregate limit

Set `concurrency_pbar` and `initial_duration_seconds` to enable domain caps:

```text
m_i = B_i / r_eff_i
p_i = clip(c * t_i - 1, concurrency_pmin, concurrency_pmax)
sum(m_i * p_i) / sum(m_i) = concurrency_pbar
cap_i = floor((1 + p_i) * m_i)
```

The monotonic clipped equation is solved by bisection plus linear refinement.
`c` has units of inverse seconds and is derived, not configured. Bounds satisfy
`0 <= pmin <= pbar <= pmax`. Before flooring the total budget is
`(1 + pbar) * sum(m_i)`; flooring loses less than one group per domain. A cap
shrink retains already running groups and prevents new starts in that domain
until its pending count is below the new cap.

`max_concurrent_samples` separately bounds aggregate submissions. With group
submission and common sibling count `K`, the aggregate cap is
`floor(max_concurrent_samples / K)` whole groups. Slots release only after the
entire group, including reward work, settles. Domain caps never replace an
explicit aggregate limit. If omitted, the aggregate cap is the sum of adaptive
domain caps, or `G` when duration caps are disabled.

**Only `submission_granularity='group'` is supported.** A value of `sample` fails
configuration validation. Sample backfill requires original-sibling completion
events across distributed agent-loop workers; native sandbox segment counts or
per-loop semaphores do not implement this contract. The 512-sample group limit
with K=16 permits 32 pending whole groups, not Miles-style partially completed
groups backfilled at sibling completion.

The adaptive engine replaces quota admission knobs such as strategy
`over_sample_threshold`, `max_pending_tasks` and `should_continue_fn` with these
scheduler-level limits. Keep the task's strategy for native
`enable_partial_rollout`, sync-cycle `max_staleness`, `max_token_staleness` and
`tail_batch_trigger_size` retry/expiry behavior. It retains the trainer's
pause/weight-publication/continue and `EXPIRED_BATCH` model-sync boundaries;
no training policy, advantage computation, or optimizer behavior changes.

## Configuration fragment

See [the importable example](../../../../examples/rl/adaptive_domain_scheduler.py)
for all fields. It supplies illustrative Cyber/SWE/TB priors and accepts already
configured generic TaskSpecs; it is not a full model/runtime recipe or an
instruction to submit a job. For example:

```python
from xtuner.v1.rl.agent_loop_manager import (
    AdaptiveDomainSchedulerConfig,
    DisaggAgentLoopManagerConfig,
)

manager_cfg = DisaggAgentLoopManagerConfig(
    tasks=tasks,  # Existing generic DisaggTaskSpecConfigs, equal prompt_repeat_k.
    domain_scheduler=AdaptiveDomainSchedulerConfig(
        target_groups={"cyber": 12, "swe": 20},
        initial_acceptance_rates={"cyber": 0.8, "swe": 0.8},
        concurrency_pbar=0.75,
        concurrency_pmax=1.5,
        initial_duration_seconds={"cyber": 600, "swe": 300},
        max_concurrent_samples=512,
        submission_granularity="group",
        group_start_interval=3,
        buffer_capacity_factor=1,
        min_trainable_siblings=12,  # Requires K >= 12 on each sampler.
    ),
)
```

Manager subclasses/wrappers must forward `domain_scheduler` when reconstructing
the base manager. Use compatible per-domain agent-loop adapters; adding
scheduling identities alone does not supply native multi-runtime routing.

## Pause, checkpoint and observability

Pause stops admissions and uses native loop-pause requests, periodic retries,
and bounded pending cleanup. On a full FIFO, drained accepted surplus cannot fit:
`unused_completed_groups='drop'` explicitly releases/discards those groups and
increments `scheduler/dropped_on_pause`. It is the only currently supported
unused-result policy. It does not cancel running groups on a cap shrink or count
the discarded group as another filter trial. Shutdown cancels outstanding work,
releases slots, and drains returned results. Fatal producer errors propagate to
`get_batch` instead of leaving the learner waiting indefinitely.

Save is rejected while adaptive pending work or waiting results remain. Existing
sampler and replay checkpoints are retained, and the domain RNG state is stored
in manager state. Fresh processes restart acceptance/duration EWMAs from priors;
finished-buffer contents follow the existing `no_save_replay_buffer` setting.
Online estimates and running work are not a fully resumable scheduler checkpoint.

`get_scheduler_metrics()` returns process-level estimates and observations.
Metrics are snapshotted into `ProduceBatchResult.scheduler_metrics` at batch
handoff and added to the trainer's normal scalar writer. Fixed scheduling returns
no scheduler metrics. Each `scheduler/<task_name>/` prefix contains target/window
groups, draw weight, filter acceptance EWMA and observation count, whole-group
duration observations, and current in-flight groups. When enabled, it also
contains duration seconds, required groups, oversampling and continuous/integer
caps. `scheduler/c_per_second`, `scheduler/aggregate_group_cap`,
`scheduler/completed_waiting_for_fifo`, and `scheduler/dropped_on_pause` describe
global state. Snapshot gauges can exceed a newly shrunken cap. They do not advance
the optimizer axis while batch collection is stalled.

CPU tests cover public manager admission/collection, filter provenance, solver
invariants, FIFO/backpressure/pacing, pause/cancel/checkpoint behavior, and fixed
mode regressions. GPU training, distributed engine execution and live sandbox
end-to-end performance require separate approved run configurations.
