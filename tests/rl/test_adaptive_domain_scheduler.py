"""Behavioral CPU tests through manager/config/filter/replay public entry points."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop import maybe_filter_invalid_sample
from xtuner.v1.rl.agent_loop_manager import (
    AdaptiveDomainSchedulerConfig,
    AgentLoopManager,
    AgentLoopManagerConfig,
    DisaggAgentLoopManager,
    DisaggAgentLoopManagerConfig,
    DisaggAsyncProduceStrategyConfig,
    FixedDomainSchedulerConfig,
    ProduceBatchStatus,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.agent_loop_manager.produce_utils import _TaskRunner
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


def config(**kwargs):
    return AdaptiveDomainSchedulerConfig(
        **{**dict(target_groups={"slow": 1, "fast": 1}, initial_acceptance_rates={"slow": 0.5, "fast": 1.0}), **kwargs}
    )


def state(uid, *, status=Status.INIT, reward=None):
    return RolloutState(
        rollout_id=uid,
        group_id=uid // 2,
        message=[{"role": "user", "content": "prompt"}],
        prompt_ids=[1],
        response="generated",
        response_ids=[2],
        response_mask=[1],
        tokens=[1, 2],
        status=status,
        reward=reward,
        extra_fields={"data_source": "native-source"},
    )


class Sampler:
    prompt_repeat_k = 2

    def __init__(self, name):
        self.name = name
        self.cursor = 0

    async def sample(self, task_name, group_status):
        self.cursor += 1
        return [state(self.cursor * 2 + i) for i in range(2)]

    def save(self, path):
        (Path(path) / "cursor").write_text(str(self.cursor))

    def resume(self, path):
        self.cursor = int((Path(path) / "cursor").read_text())

    def __len__(self):
        return 100


class Loop:
    """Deliberately no .logger (Ray/router proxies need not expose one)."""

    def __init__(self, name, *, delay=0.005, filter_fn=None, gate=None, outcomes=()):
        self.name = name
        self.delay = delay
        self.filter_fn = filter_fn
        self.gate = gate
        self.started = []
        self.finished = 0
        self.active = 0
        self.max_active = 0
        self.cancelled = 0
        self.outcomes = iter(outcomes)

    async def generate_group(self, group, **kwargs):
        self.started.append(asyncio.get_running_loop().time())
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(self.delay)
            outcome = next(self.outcomes, "pass")
            if outcome == "exception":
                raise RuntimeError("generation failed")
            for item in group:
                item.status = Status.FAILED if outcome == "infra" else Status.COMPLETED
                item.reward = None if outcome == "missing" else {"score": float(item.rollout_id % 2)}
            self.finished += 1
            return maybe_filter_invalid_sample(group, self.filter_fn, MagicMock())
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1

    async def pause(self):
        if self.gate is not None:
            self.gate.set()


def manager(*, adaptive=None, disagg=False, loops=None, strategy=None):
    loops = loops or [Loop("slow"), Loop("fast")]
    replay = AsyncReplayBufferConfig().build()
    runners = []
    for order, loop in enumerate(loops):
        strategy_cfg = strategy or (
            DisaggAsyncProduceStrategyConfig(max_staleness=20) if disagg else SyncProduceStrategyConfig()
        )
        runners.append(
            _TaskRunner(
                task_name=loop.name,
                agent_loop=loop,
                produce_strategy=strategy_cfg.build(),
                sampler=Sampler(loop.name),
                weight=1,
                order=order,
            )
        )
    ctl = MagicMock()
    ctl.pause_generation.remote = AsyncMock()
    ctl.continue_generation.remote = AsyncMock()
    cls = DisaggAgentLoopManager if disagg else AgentLoopManager
    return cls(runners, replay, ctl, domain_scheduler=adaptive), loops


async def until(predicate, timeout=2):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout)


async def stop(mgr, task):
    mgr.shutdown()
    await asyncio.wait_for(task, 2)


def test_config_choice_defaults_and_validated_limits():
    task = TaskSpecConfig.model_construct(
        task_name="slow",
        agent_loop_config=MagicMock(),
        sampler_config=MagicMock(),
    )
    assert isinstance(AgentLoopManagerConfig(tasks=task).domain_scheduler, FixedDomainSchedulerConfig)
    cfg = AgentLoopManagerConfig(tasks=task, domain_scheduler=config())
    assert cfg.domain_scheduler.kind == "adaptive"
    assert isinstance(
        DisaggAgentLoopManagerConfig.model_construct(tasks=[]).domain_scheduler, FixedDomainSchedulerConfig
    )
    for fields in (
        {"submission_granularity": "sample"},
        {"target_groups": {"slow": 0, "fast": 2}},
        {"initial_acceptance_rates": {"slow": 0.5}},
        {"initial_acceptance_rates": {"slow": float("nan"), "fast": 1}},
        {"acceptance_floor": 1e-320, "initial_acceptance_rates": {"slow": 1, "fast": 1}},
        {"initial_duration_seconds": {"slow": 1, "fast": 1}},
        {"concurrency_pbar": 0.5},
        {"group_start_interval": float("inf")},
        {"concurrency_pbar": 2, "initial_duration_seconds": {"slow": 1, "fast": 1}},
    ):
        with pytest.raises(ValidationError):
            config(**fields)
    with pytest.raises(ValueError, match="whole sibling group"):
        manager(adaptive=config(max_concurrent_samples=1))
    with pytest.raises(ValueError, match="exactly"):
        manager(adaptive=config(target_groups={"other": 2}, initial_acceptance_rates={"other": 0.5}))


@pytest.mark.asyncio
async def test_colocated_public_batch_soft_mix_fifo_and_source_preserved():
    mgr, loops = manager(adaptive=config(seed=0, max_concurrent_samples=4))
    with pytest.raises(ValueError, match="sum"):
        await mgr.produce_batch(3, 1, model_step=0)
    result = await asyncio.wait_for(mgr.produce_batch(2, 1, model_step=0), 2)
    assert len(result.rollout_states) == 2
    assert all(
        item.extra_fields["data_source"] == "native-source" for group in result.rollout_states for item in group
    )
    assert all(
        item.extra_fields["_domain_scheduler"]["domain"] in ("slow", "fast")
        for group in result.rollout_states
        for item in group
    )
    assert sum(result.task_batch_sizes.values()) == 2
    assert not any(loop.active for loop in loops)
    assert (
        sum(mgr.get_scheduler_metrics()[f"scheduler/{name}/acceptance_observations"] for name in ("slow", "fast")) == 0
    )
    assert sum(mgr.get_scheduler_metrics()[f"scheduler/{name}/in_flight_groups"] for name in ("slow", "fast")) == 0


@pytest.mark.asyncio
async def test_disagg_public_fifo_batch_ignores_exact_task_quotas_and_bounded_window():
    cfg = config(max_concurrent_samples=4, buffer_capacity_factor=2)
    mgr, _ = manager(adaptive=cfg, disagg=True)
    for name in ("fast", "fast", "slow", "slow"):
        group = [state(10, status=Status.COMPLETED, reward={"score": 1})]
        await mgr.replay_buffer.put(group, name, model_step=0, current_train_step=1)
    count, counts = await mgr.replay_buffer.completed_fifo_snapshot(mgr.task_names, 2)
    assert count == 4 and counts == {"slow": 0, "fast": 2}
    result = await asyncio.wait_for(mgr.get_batch(2, 1), 2)
    assert result.task_batch_sizes == {"slow": 0, "fast": 2}
    count, counts = await mgr.replay_buffer.completed_fifo_snapshot(mgr.task_names, 2)
    assert count == 2 and counts == {"slow": 2, "fast": 0}
    # Handoff does not reserve a partial batch or keep earlier-batch counts.
    assert mgr._produce_progress.consumed_samples == {"slow": 0, "fast": 2}


@pytest.mark.asyncio
async def test_global_pacing_collects_completed_groups_before_next_admission():
    mgr, loops = manager(adaptive=config(group_start_interval=0.15, max_concurrent_samples=8), disagg=True)
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await until(lambda: sum(loop.finished for loop in loops) >= 1)
        await asyncio.sleep(0.02)
        assert sum(len(loop.started) for loop in loops) == 1
        count, _ = await mgr.replay_buffer.completed_fifo_snapshot(mgr.task_names, 2)
        assert count == 1  # collection is not delayed by the pacing deadline
        result = await asyncio.wait_for(mgr.get_batch(2, 1), 2)
        assert len(result.rollout_states) == 2
        starts = sorted(t for loop in loops for t in loop.started)
        assert all(b - a >= 0.14 for a, b in zip(starts, starts[1:]))
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_group_slots_remain_held_until_whole_group_and_buffer_backpressure_drains():
    gate = asyncio.Event()
    loops = [Loop("slow", gate=gate), Loop("fast", gate=gate)]
    mgr, _ = manager(adaptive=config(max_concurrent_samples=5, buffer_capacity_factor=1), disagg=True, loops=loops)
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await until(lambda: sum(len(loop.started) for loop in loops) == 2)
        await asyncio.sleep(0.03)
        assert sum(loop.active for loop in loops) == 2  # floor(5/2)=2 groups
        with pytest.raises(RuntimeError, match="pending"):
            await mgr.save("/tmp/xtuner-adaptive-test-active-checkpoint", 0)
        gate.set()
        await until(lambda: sum(loop.finished for loop in loops) >= 2)
        await asyncio.sleep(0.03)
        count, _ = await mgr.replay_buffer.completed_fifo_snapshot(mgr.task_names, 2)
        assert count == 2
        started = sum(len(loop.started) for loop in loops)
        await asyncio.sleep(0.05)
        assert sum(len(loop.started) for loop in loops) == started
        assert sum(mgr.get_scheduler_metrics()[f"scheduler/{name}/in_flight_groups"] for name in ("slow", "fast")) == 0
        assert len((await mgr.get_batch(2, 1)).rollout_states) == 2
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_filter_verdicts_missing_rewards_infra_disabled_and_duration_are_distinct():
    loops = [
        Loop("slow", filter_fn=lambda group: False, outcomes=["missing", "infra", "pass"]),
        Loop("fast", filter_fn=None),
    ]
    mgr, _ = manager(
        adaptive=config(
            concurrency_pbar=0, initial_duration_seconds={"slow": 10, "fast": 10}, max_concurrent_samples=4
        ),
        disagg=True,
        loops=loops,
    )
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        for step in range(1, 4):
            await asyncio.wait_for(mgr.get_batch(2, step), 2)
        await mgr.pause_produce()
        metrics = mgr.get_scheduler_metrics()
        assert metrics["scheduler/slow/duration_observations"] >= 3
        assert metrics["scheduler/slow/acceptance_observations"] == metrics["scheduler/slow/duration_observations"] - 2
        assert metrics["scheduler/fast/acceptance_observations"] == 0
        assert metrics["scheduler/fast/acceptance_ewma"] == 1
        assert metrics["scheduler/slow/acceptance_ewma"] < 0.5
        assert metrics["scheduler/slow/duration_seconds"] < 10
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_pause_and_shutdown_cleanup_and_checkpoint_rng(tmp_path):
    gate = asyncio.Event()
    mgr, loops = manager(
        adaptive=config(max_concurrent_samples=8),
        disagg=True,
        loops=[Loop("slow", gate=gate), Loop("fast", gate=gate)],
    )
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await until(lambda: sum(loop.active for loop in loops) == 4)
        await asyncio.wait_for(mgr.pause_produce(), 2)
        assert not any(loop.active for loop in loops)
        starts = sum(len(loop.started) for loop in loops)
        await asyncio.sleep(0.02)
        assert sum(len(loop.started) for loop in loops) == starts
        await mgr.save(tmp_path, 0)
        resumed, _ = manager(adaptive=config(max_concurrent_samples=8), disagg=True)
        assert await resumed.resume(tmp_path) == 0
        assert resumed._adaptive.state.rng.getstate() == mgr._adaptive.state.rng.getstate()
        assert resumed._adaptive.state.acceptance == config().initial_acceptance_rates
        await mgr.continue_produce(1)
    finally:
        await stop(mgr, producer)
    assert not any(loop.active for loop in loops)


@pytest.mark.asyncio
async def test_generation_exception_is_reported_and_does_not_update_duration_or_leak_slots():
    mgr, loops = manager(
        adaptive=config(max_concurrent_samples=2),
        disagg=True,
        loops=[Loop("slow", outcomes=["exception"]), Loop("fast", outcomes=["exception"])],
    )
    producer = asyncio.create_task(mgr.produce_loop(2))
    with pytest.raises(RuntimeError, match="generation failed"):
        await asyncio.wait_for(mgr.get_batch(2, 1), 2)
    with pytest.raises(RuntimeError, match="generation failed"):
        await producer
    metrics = mgr.get_scheduler_metrics()
    assert not any(loop.active for loop in loops)
    assert sum(metrics[f"scheduler/{name}/duration_observations"] for name in ("slow", "fast")) == 0
    assert sum(metrics[f"scheduler/{name}/in_flight_groups"] for name in ("slow", "fast")) == 0


def test_filter_admission_counts_original_siblings_not_expanded_segments():
    group = [state(i, status=Status.COMPLETED, reward={"score": i % 2}) for i in range(4)]
    for item in group:
        item.extra_fields["_domain_scheduler"] = {
            "domain": "slow",
            "original_sibling_id": 0,
            "min_trainable_siblings": 2,
        }
    filter_fn = MagicMock(return_value=True)
    result = maybe_filter_invalid_sample(group, filter_fn, MagicMock())
    assert all(item.status == Status.FAILED for item in result)
    filter_fn.assert_not_called()
    assert all(item.extra_fields["_domain_scheduler"]["rejection"] == "admission" for item in result)


@pytest.mark.asyncio
async def test_native_stale_drop_does_not_add_filter_observations():
    mgr, _ = manager(
        adaptive=config(),
        disagg=True,
        strategy=DisaggAsyncProduceStrategyConfig(max_staleness=0, tail_batch_trigger_size=-1),
    )
    for name in ("slow", "fast"):
        await mgr.replay_buffer.put([state(1, status=Status.COMPLETED, reward={"score": 1})], name, model_step=0)
    # Consumer at step 3 expires data using native sync-cycle thresholds.
    read = asyncio.create_task(mgr.get_batch(2, 3))
    await asyncio.sleep(0.03)
    assert not read.done()
    assert (await mgr.replay_buffer.completed_fifo_snapshot(mgr.task_names, 2))[0] == 0
    assert (
        sum(mgr.get_scheduler_metrics()[f"scheduler/{name}/acceptance_observations"] for name in ("slow", "fast")) == 0
    )
    mgr.shutdown()
    assert not (await read).rollout_states


@pytest.mark.asyncio
async def test_cap_shrink_keeps_running_groups_and_stops_new_domain_admissions():
    slow_gates = [asyncio.Event() for _ in range(4)]
    fast_gate = asyncio.Event()

    class GatedLoop(Loop):
        async def generate_group(self, group, **kwargs):
            index = group[0].group_id - 1
            if self.name == "slow":
                await slow_gates[index].wait()
            return await super().generate_group(group, **kwargs)

    loops = [GatedLoop("slow", filter_fn=lambda _: True), Loop("fast", gate=fast_gate)]
    mgr, _ = manager(
        adaptive=config(
            initial_acceptance_rates={"slow": 0.25, "fast": 1},
            acceptance_ewma_alpha=1,
            concurrency_pbar=0,
            initial_duration_seconds={"slow": 1, "fast": 1},
            max_concurrent_samples=10,
        ),
        disagg=True,
        loops=loops,
    )
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await until(
            lambda: mgr.get_scheduler_metrics()["scheduler/slow/in_flight_groups"] == 4
            and mgr.get_scheduler_metrics()["scheduler/fast/in_flight_groups"] == 1
        )
        assert mgr.task_runners[0].sampler.cursor == 4
        slow_gates[0].set()
        await until(
            lambda: mgr.get_scheduler_metrics()["scheduler/slow/group_cap"] == 1
            and mgr.get_scheduler_metrics()["scheduler/slow/in_flight_groups"] == 3
        )
        assert mgr.get_scheduler_metrics()["scheduler/slow/in_flight_groups"] == 3
        await asyncio.sleep(0.03)
        assert mgr.task_runners[0].sampler.cursor == 4
        assert loops[0].cancelled == 0
    finally:
        for event in slow_gates:
            event.set()
        fast_gate.set()
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_returned_duration_excludes_fifo_backpressure_and_pause_drops_surplus():
    gate = asyncio.Event()
    loops = [Loop("slow", gate=gate), Loop("fast", gate=gate)]
    mgr, _ = manager(
        adaptive=config(
            max_concurrent_samples=6,
            buffer_capacity_factor=1,
            concurrency_pbar=1,
            initial_acceptance_rates={"slow": 1, "fast": 1},
            initial_duration_seconds={"slow": 10, "fast": 10},
            duration_ewma_alpha=1,
        ),
        disagg=True,
        loops=loops,
    )
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await until(lambda: sum(loop.active for loop in loops) == 3)
        gate.set()
        await until(lambda: mgr.get_scheduler_metrics()["scheduler/completed_waiting_for_fifo"] == 1)
        before = mgr.get_scheduler_metrics()
        await asyncio.sleep(0.1)
        after = mgr.get_scheduler_metrics()
        assert sum(after[f"scheduler/{name}/duration_observations"] for name in ("slow", "fast")) == 3
        assert after["scheduler/slow/duration_seconds"] == before["scheduler/slow/duration_seconds"]
        assert after["scheduler/fast/duration_seconds"] == before["scheduler/fast/duration_seconds"]
        assert sum(loop.active for loop in loops) == 0
        assert (await mgr.replay_buffer.completed_fifo_snapshot(mgr.task_names, 2))[0] == 2
        await mgr.pause_produce()
        assert mgr.get_scheduler_metrics()["scheduler/dropped_on_pause"] == 1
        assert mgr.get_scheduler_metrics()["scheduler/completed_waiting_for_fifo"] == 0
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_pause_winning_a_sampler_race_returns_prompt_to_retry_pool():
    sampling = asyncio.Event()
    release = asyncio.Event()
    mgr, loops = manager(adaptive=config(max_concurrent_samples=2), disagg=True)
    for runner in mgr.task_runners:
        original_sample = runner.sampler.sample

        async def blocking_sample(*args, _sample=original_sample, **kwargs):
            group = await _sample(*args, **kwargs)
            sampling.set()
            await release.wait()
            return group

        runner.sampler.sample = blocking_sample
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await sampling.wait()
        pause = asyncio.create_task(mgr.pause_produce())
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.wait_for(pause, 2)
        assert sum(len(loop.started) for loop in loops) == 0
        assert (
            sum(await asyncio.gather(*[mgr.replay_buffer.count(name, Status.ABORTED) for name in mgr.task_names])) == 1
        )
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_sandbox_provenance_survives_expansion_and_failed_sibling_drop():
    import contextvars
    import sys
    import types

    from xtuner.v1.data_proto.rl_data import SampleParams

    # Match the functional optional-lagent stand-ins used by test_sandbox_pool.
    # This exercises sandbox group bookkeeping without starting any sandbox.
    lagent = types.ModuleType("lagent")
    utils = types.ModuleType("lagent.utils")
    utils.create_object = lambda value: value
    utils.ctx_session_id = contextvars.ContextVar("session_id", default=None)
    rate_limiter = types.ModuleType("lagent.utils.rate_limiter")
    rate_limiter.get_shared_async_token_bucket = lambda *args, **kwargs: None
    sys.modules.setdefault("lagent", lagent)
    sys.modules.setdefault("lagent.utils", utils)
    sys.modules.setdefault("lagent.utils.rate_limiter", rate_limiter)
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop

    loop = object.__new__(AgentInSandboxLoop)
    loop.sample_params = SampleParams()
    loop._sample_semaphore = None
    loop.mode = "train"
    loop.logger = MagicMock()
    loop.is_valid_sample_fn = MagicMock(return_value=True)
    loop._teacher_scorer = SimpleNamespace(
        on_sample_ready=AsyncMock(side_effect=lambda sample: sample),
        on_group_ready=AsyncMock(side_effect=lambda group: group),
    )

    async def generate_sample(sample):
        copies = [sample.model_copy(deep=True) for _ in range(3 if sample.rollout_id == 0 else 1)]
        for item in copies:
            item.status = Status.COMPLETED if sample.rollout_id == 0 else Status.FAILED
            item.reward = {"score": 1}
        return copies

    loop.generate_sample = generate_sample
    group = [state(i) for i in range(2)]
    for item in group:
        item.extra_fields["_domain_scheduler"] = {
            "domain": "slow",
            "original_group_id": 1,
            "original_sibling_id": item.rollout_id,
            "min_trainable_siblings": 2,
        }
    result = await loop.generate_group(group)
    assert len(result) == 3
    assert all(item.status == Status.FAILED for item in result)
    assert {item.extra_fields["_domain_scheduler"]["original_sibling_id"] for item in result} == {0}
    loop.is_valid_sample_fn.assert_not_called()


@pytest.mark.asyncio
async def test_fast_domain_fills_public_batch_while_slow_domain_is_running():
    slow_gate = asyncio.Event()
    loops = [Loop("slow", gate=slow_gate), Loop("fast")]
    mgr, _ = manager(
        adaptive=config(
            target_groups={"slow": 1, "fast": 3},
            initial_acceptance_rates={"slow": 1, "fast": 1},
            concurrency_pbar=0.5,
            concurrency_pmax=1,
            initial_duration_seconds={"slow": 10, "fast": 1},
            max_concurrent_samples=12,
        ),
        disagg=True,
        loops=loops,
    )
    producer = asyncio.create_task(mgr.produce_loop(4))
    try:
        result = await asyncio.wait_for(mgr.get_batch(4, 1), 2)
        assert len(result.rollout_states) == 4
        assert result.task_batch_sizes == {"slow": 0, "fast": 4}
        assert loops[0].active > 0
        assert (
            mgr.get_scheduler_metrics()["scheduler/slow/in_flight_groups"]
            <= mgr.get_scheduler_metrics()["scheduler/slow/group_cap"]
        )
    finally:
        slow_gate.set()
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_pause_timeout_cancels_unreturned_attempts_without_estimate_observations():
    class StubbornLoop(Loop):
        async def pause(self):
            pass

    gate = asyncio.Event()
    loops = [StubbornLoop("slow", gate=gate), StubbornLoop("fast", gate=gate)]
    mgr, _ = manager(adaptive=config(max_concurrent_samples=4), disagg=True, loops=loops)
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        await until(lambda: sum(loop.active for loop in loops) == 2)
        with patch("xtuner.v1.rl.agent_loop_manager.adaptive_producer.PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S", 0.02):
            await asyncio.wait_for(mgr.pause_produce(), 1)
        assert sum(loop.active for loop in loops) == 0
        assert sum(loop.cancelled for loop in loops) == 2
        metrics = mgr.get_scheduler_metrics()
        assert sum(metrics[f"scheduler/{name}/duration_observations"] for name in ("slow", "fast")) == 0
        assert sum(metrics[f"scheduler/{name}/in_flight_groups"] for name in ("slow", "fast")) == 0
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_wave", [False, True])
async def test_native_expired_batch_model_sync_boundary_survives_adaptive_admission(oversized_wave):
    from xtuner.v1.rl.agent_loop_manager import AgentLoopManagerStatus

    mgr, loops = manager(
        adaptive=config(max_concurrent_samples=8 if oversized_wave else 4),
        disagg=True,
        strategy=DisaggAsyncProduceStrategyConfig(max_staleness=0),
    )
    producer = asyncio.create_task(mgr.produce_loop(2))
    try:
        if oversized_wave:
            await until(lambda: sum(loop.finished for loop in loops) >= 4)
            await until(lambda: mgr._status == AgentLoopManagerStatus.EXPIRED_BATCH)
            assert mgr._produce_progress.producer_future_step == 2
        assert len((await asyncio.wait_for(mgr.get_batch(2, 1), 2)).rollout_states) == 2
        await until(lambda: mgr._status == AgentLoopManagerStatus.EXPIRED_BATCH)
        expired = await asyncio.wait_for(mgr.get_batch(2, 2), 2)
        assert expired.status == ProduceBatchStatus.EXPIRED_BATCH and not expired.rollout_states
        await mgr.pause_produce()
        await mgr.continue_produce(model_step=1)
        assert len((await asyncio.wait_for(mgr.get_batch(2, 2), 2)).rollout_states) == 2
    finally:
        await stop(mgr, producer)


@pytest.mark.asyncio
async def test_adaptive_targets_enable_domains_even_with_zero_legacy_weights():
    from dataclasses import replace

    original, _ = manager()
    tasks = [replace(task, weight=0) for task in original.task_runners]
    with pytest.raises(ValueError, match="weight"):
        AgentLoopManager(tasks, original.replay_buffer, original._rollout_controller)
    adaptive = AgentLoopManager(tasks, original.replay_buffer, original._rollout_controller, domain_scheduler=config())
    assert len((await asyncio.wait_for(adaptive.produce_batch(2, 1, model_step=0), 2)).rollout_states) == 2


@pytest.mark.parametrize("invalid", ["empty_response", "empty_ids", "masked", "agentic_no_labels", "agentic_masked"])
def test_no_trainable_token_is_admission_failure_before_rl_filter(invalid):
    group = [state(i, status=Status.COMPLETED, reward={"score": i % 2}) for i in range(2)]
    for item in group:
        item.extra_fields["_domain_scheduler"] = {
            "domain": "slow",
            "original_sibling_id": item.rollout_id,
            "min_trainable_siblings": 2,
        }
    if invalid == "empty_response":
        group[0].response = ""
    elif invalid == "empty_ids":
        group[0].response_ids = []
    elif invalid == "masked":
        group[0].response_mask = [0]
    else:
        group[0].input_ids = [1, 2]
        group[0].labels = None if invalid == "agentic_no_labels" else [-100, -100]
    filter_fn = MagicMock(return_value=True)
    result = maybe_filter_invalid_sample(group, filter_fn, MagicMock())
    filter_fn.assert_not_called()
    assert all(item.status == Status.FAILED for item in result)
    assert all(item.extra_fields["_domain_scheduler"]["rejection"] == "admission" for item in result)


@pytest.mark.asyncio
async def test_sampler_failure_releases_reserved_slots_and_reaches_consumer():
    mgr, loops = manager(adaptive=config(max_concurrent_samples=2), disagg=True)
    for runner in mgr.task_runners:
        runner.sampler.sample = AsyncMock(side_effect=RuntimeError("sampler failed"))
    producer = asyncio.create_task(mgr.produce_loop(2))
    with pytest.raises(RuntimeError, match="sampler failed"):
        await asyncio.wait_for(mgr.get_batch(2, 1), 2)
    with pytest.raises(RuntimeError, match="sampler failed"):
        await producer
    assert sum(loop.active for loop in loops) == 0
    metrics = mgr.get_scheduler_metrics()
    assert sum(metrics[f"scheduler/{name}/in_flight_groups"] for name in ("slow", "fast")) == 0
    assert sum(metrics[f"scheduler/{name}/duration_observations"] for name in ("slow", "fast")) == 0
