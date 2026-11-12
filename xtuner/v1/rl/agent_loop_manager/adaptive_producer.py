"""Shared admission/collection engine for the opt-in domain scheduler."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable

from xtuner.v1.data_proto.rl_data import RolloutState, Status, get_group_status
from xtuner.v1.rl.agent_loop.agent_loop import maybe_filter_invalid_sample
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.trace_store import release_and_discard_rollout_groups
from xtuner.v1.rl.utils import PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S, cancel_and_drain, create_task
from xtuner.v1.utils import get_logger

from .domain_scheduler import AdaptiveDomainSchedulerConfig, DomainSchedulingState
from .produce_utils import PERIODIC_ABORT_INTERVAL_S, BaseProduceContext, _TaskRunner, request_agent_loop_pause


class AdaptiveDomainProducer:
    """One global group admission stream and one completion FIFO writer.

    Running groups retain independent domain and aggregate slots. Completed
    results may wait outside the bounded replay FIFO (at most one admission
    wave); backpressure stops further starts, while failures and timings are
    still collected. Pause drops accepted surplus after draining generation.
    """

    def __init__(self, config: AdaptiveDomainSchedulerConfig, tasks: list[_TaskRunner], replay_buffer: ReplayBuffer):
        self.config = config
        self.tasks = {task.task_name: task for task in tasks}
        if self.tasks.keys() != config.target_groups.keys():
            raise ValueError("Adaptive target_groups must cover exactly the manager's task_name identities.")
        sibling_counts = {task.sampler.prompt_repeat_k for task in tasks}
        if len(sibling_counts) != 1 or min(sibling_counts) <= 0:
            raise ValueError(
                "Adaptive scheduling currently requires the same positive prompt_repeat_k for every task."
            )
        self.siblings = next(iter(sibling_counts))
        if config.min_trainable_siblings > self.siblings:
            raise ValueError("min_trainable_siblings cannot exceed prompt_repeat_k.")
        if config.max_concurrent_samples is not None and config.max_concurrent_samples < self.siblings:
            raise ValueError("max_concurrent_samples must fit at least one whole sibling group.")
        self.replay_buffer = replay_buffer
        self.state = DomainSchedulingState(config)
        self._pending: dict[asyncio.Task, BaseProduceContext] = {}
        self._completed: asyncio.Queue[asyncio.Task] = asyncio.Queue()
        self._ready: deque[tuple[BaseProduceContext, list[RolloutState]]] = deque()
        self._collect_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._last_start: float | None = None
        self._pausing = False
        self._sampling = False
        self.dropped_on_pause = 0

    def validate_batch_size(self, batch_size: int) -> None:
        if batch_size != self.config.batch_size:
            raise ValueError(
                f"Adaptive target_groups sum to {self.config.batch_size}, requested batch_size={batch_size}."
            )

    def pending_task_count(self) -> int:
        return len(self._pending) + int(self._sampling)

    def buffered_result_count(self) -> int:
        return len(self._ready)

    def aggregate_group_cap(self) -> int:
        if self.config.max_concurrent_samples is not None:
            return self.config.max_concurrent_samples // self.siblings
        if self.state.solution is not None:
            return sum(self.state.solution.group_caps.values())
        return self.config.batch_size

    async def _generate(self, ctx: BaseProduceContext, group: list[RolloutState]) -> list[RolloutState]:
        started = time.perf_counter()
        try:
            result = await ctx.generate_group(
                group,
                enable_partial_rollout=getattr(
                    self.tasks[ctx.task_name].produce_strategy, "enable_partial_rollout", False
                ),
            )
            # Existing loops call maybe_filter_invalid_sample before their
            # Teacher hook. Do not invoke the configured filter a second time.
            # This no-filter guard also catches adapters which omitted that hook.
            result = maybe_filter_invalid_sample(result, None, get_logger())
            verdict = None
            if result:
                provenance = result[0].extra_fields.get("_domain_scheduler", {})
                verdict = provenance.get("filter_verdict")
                if provenance.get("rejection") is not None:
                    verdict = None
            self.state.observe(
                ctx.task_name,
                duration_seconds=time.perf_counter() - started,
                filter_verdict=verdict,
            )
            return result
        finally:
            self._changed.set()

    def _on_done(self, task: asyncio.Task, domain: str) -> None:
        # A task cancelled before its coroutine first runs must release slots too.
        self.state.in_flight[domain] -= 1
        self._completed.put_nowait(task)
        self._changed.set()

    async def collect(self, *, propagate_errors: bool = True) -> None:
        """Claim completions once, including while pacing or FIFO-backpressured."""
        async with self._collect_lock:
            while not self._completed.empty():
                task = self._completed.get_nowait()
                ctx = self._pending.pop(task, None)
                if ctx is None or task.cancelled():
                    continue
                try:
                    group = task.result()  # Preserve generation exceptions as producer errors.
                except Exception:
                    if propagate_errors:
                        raise
                    continue
                if group:
                    self._ready.append((ctx, group))
            count, _ = await self.replay_buffer.completed_fifo_snapshot(list(self.tasks), self.config.batch_size)
            remaining = deque()
            while self._ready:
                ctx, group = self._ready.popleft()
                if get_group_status(group) == Status.COMPLETED and count >= self.config.buffer_capacity:
                    remaining.append((ctx, group))
                    continue
                if await ctx.put_generated_group(group):
                    count += 1
            self._ready = remaining

    async def admit_one(
        self,
        context_for: Callable[[str], BaseProduceContext],
        should_abort: Callable[[], bool],
    ) -> bool:
        """Try one globally paced admission; never advance capped domain cursors."""
        async with self._admission_lock:
            if self._pausing or should_abort():
                return False
            await self.collect()
            count, prefix = await self.replay_buffer.completed_fifo_snapshot(list(self.tasks), self.config.batch_size)
            if self._ready or count >= self.config.buffer_capacity:
                return False
            if sum(self.state.in_flight.values()) >= self.aggregate_group_cap():
                return False
            now = time.perf_counter()
            if self._last_start is not None and now - self._last_start < self.config.group_start_interval:
                return False
            self.state.refresh_caps()
            eligible = [
                name
                for name in self.tasks
                if self.state.solution is None or self.state.in_flight[name] < self.state.solution.group_caps[name]
            ]
            domain = self.state.draw(prefix, eligible)
            if domain is None:
                return False
            ctx = context_for(domain)
            task = self.tasks[domain]
            # Reserve both domain and aggregate slots before sampler/replay
            # awaits. A concurrent completion may shrink caps while sampling;
            # this already admitted reservation then follows cap-shrink rules.
            self.state.in_flight[domain] += 1
            self._sampling = True
            spawned = False
            try:
                expired_count = await ctx.expired_count()
                trigger = getattr(task.produce_strategy, "tail_batch_trigger_size", -1)
                group = await ctx.sample_group(
                    from_expired_pool=trigger >= 0 and expired_count > 0 and expired_count >= trigger
                )
                if self._pausing or should_abort():
                    # Return a prompt sampled across a weight-publication pause
                    # to the native retry pool, without launching generation.
                    for item in group:
                        item.status = Status.ABORTED
                    await self.replay_buffer.put(group, domain)
                    return False
                for item in group:
                    item.extra_fields["_domain_scheduler"] = {
                        "domain": domain,
                        "original_group_id": item.group_id,
                        "original_sibling_id": item.rollout_id,
                        "min_trainable_siblings": self.config.min_trainable_siblings,
                    }
                self._last_start = time.perf_counter()
                future = create_task(self._generate(ctx, group))
                self._pending[future] = ctx
                future.add_done_callback(lambda done: self._on_done(done, domain))
                spawned = True
                return True
            finally:
                self._sampling = False
                if not spawned:
                    self.state.in_flight[domain] -= 1
                    self._changed.set()

    async def wait_for_change(self, timeout: float = 0.05) -> None:
        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    async def pause(self, context_for: Callable[[str], BaseProduceContext]) -> float:
        """Stop admissions, request native pause and bound pending cleanup time."""
        started = time.perf_counter()
        self._pausing = True
        # Wait for a sampler operation already in progress to return its prompt.
        async with self._admission_lock:
            pass
        pause_tasks = [
            create_task(request_agent_loop_pause(context_for(name), pending_count=self.pending_task_count()))
            for name in self.tasks
        ]
        next_abort = started + PERIODIC_ABORT_INTERVAL_S
        try:
            while self._pending and time.perf_counter() - started < PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
                await self.collect()
                if self._pending:
                    if PERIODIC_ABORT_INTERVAL_S > 0 and time.perf_counter() >= next_abort:
                        pause_tasks.extend(
                            create_task(
                                request_agent_loop_pause(context_for(name), pending_count=self.pending_task_count())
                            )
                            for name in self.tasks
                            if self.state.in_flight[name] > 0
                        )
                        next_abort = time.perf_counter() + PERIODIC_ABORT_INTERVAL_S
                    await self.wait_for_change()
            if self._pending:
                await cancel_and_drain(list(self._pending))
                await self.collect()
            await self.collect()
            # A bounded FIFO cannot absorb an arbitrarily large completion wave
            # during pause. These already-observed accepted groups are unused.
            while self._ready:
                ctx, group = self._ready.popleft()
                await release_and_discard_rollout_groups([group])
                self.dropped_on_pause += 1
        finally:
            await cancel_and_drain(pause_tasks)
        return time.perf_counter() - started

    async def cancel_pending(self) -> None:
        self._pausing = True
        await cancel_and_drain(list(self._pending))
        await self.collect(propagate_errors=False)
        while self._ready:
            _, group = self._ready.popleft()
            await release_and_discard_rollout_groups([group])

    def continue_produce(self) -> None:
        self._pausing = False
        # No catch-up credits accumulate across pauses/backpressure.
        self._changed.set()

    def metrics(self) -> dict[str, float]:
        return {
            **self.state.metrics(),
            "scheduler/dropped_on_pause": self.dropped_on_pause,
            "scheduler/completed_waiting_for_fifo": len(self._ready),
            "scheduler/aggregate_group_cap": self.aggregate_group_cap(),
        }
