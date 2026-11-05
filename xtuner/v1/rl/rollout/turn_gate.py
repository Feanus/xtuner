"""Request-boundary admission for colocated agentic rollout.

The gate never interrupts healthy generations to publish weights. A ticket owns
its version until the response *and* trace commit finish. Cancellation is a
separate operation: callers must abort the mapped engine session and wait for
ticket completion before allowing GPU resource changes.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Callable
from uuid import uuid4


class TurnCancelled(RuntimeError):
    """A terminal logical-session cancellation, never a format correction."""


@dataclass
class TurnTicket:
    request_id: str
    session_id: str
    engine_session_id: int
    version: int
    source_optimizer_step: int
    barrier_epoch: int
    enqueued_at: float
    admitted_at: float
    phase: str = "generating"
    cancel_reason: str | None = None
    trace_committed: bool = False


class TurnGate:
    def __init__(self, max_active: int = 16, max_waiting: int = 128):
        if min(max_active, max_waiting) <= 0:
            raise ValueError("turn gate limits must be positive")
        self.max_active = max_active
        self.instance_id = uuid4().hex
        self.max_waiting = max_waiting
        self.condition = asyncio.Condition()
        self.accepting = False
        self.closed = False
        self.epoch = 0
        self.version = -1
        self.source_optimizer_step = -1
        self.prepared: tuple[int, int, int] | None = None
        self.active: dict[str, TurnTicket] = {}
        self.waiting: dict[str, str] = {}
        self.cancelled: dict[str, str] = {}

    def status(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "open": self.accepting, "closed": self.closed, "epoch": self.epoch,
            "version": self.version, "source_optimizer_step": self.source_optimizer_step,
            "active": len(self.active), "committing": sum(t.phase == "committing" for t in self.active.values()),
            "waiting": len(self.waiting), "requests": [asdict(t) for t in self.active.values()],
        }

    async def close_admission(self, epoch: int) -> dict:
        async with self.condition:
            if epoch < self.epoch:
                raise ValueError("stale barrier epoch")
            self.accepting = False
            self.epoch = epoch
            self.prepared = None
            self.condition.notify_all()
            return self.status()

    async def prepare_version(self, version: int, source_optimizer_step: int, epoch: int) -> dict:
        async with self.condition:
            if self.closed or self.accepting or self.active:
                raise RuntimeError("version prepare requires a live, closed, drained gate")
            if epoch != self.epoch or version < self.version or source_optimizer_step < self.source_optimizer_step:
                raise ValueError("invalid version publication")
            self.prepared = (version, source_optimizer_step, epoch)
            return self.status()

    async def open_admission(self, version: int, source_optimizer_step: int, epoch: int) -> dict:
        async with self.condition:
            publication = (version, source_optimizer_step, epoch)
            if self.closed or self.active or self.prepared != publication:
                raise RuntimeError("all replicas must prepare the publication before open")
            self.version, self.source_optimizer_step, self.epoch = publication
            self.accepting = True
            self.condition.notify_all()
            return self.status()

    async def cancel_session(self, session_id: str, reason: str) -> list[dict]:
        async with self.condition:
            self.cancelled[session_id] = reason
            tickets = [t for t in self.active.values() if t.session_id == session_id]
            for ticket in tickets:
                ticket.cancel_reason = reason
            self.condition.notify_all()
            return [asdict(t) for t in tickets]

    async def shutdown(self) -> list[dict]:
        async with self.condition:
            self.closed = True
            self.accepting = False
            for ticket in self.active.values():
                ticket.cancel_reason = "shutdown"
            self.condition.notify_all()
            return [asdict(t) for t in self.active.values()]

    async def drain(self, timeout: float) -> dict:
        if self.accepting:
            raise RuntimeError("close admission before drain")
        async with asyncio.timeout(timeout):
            async with self.condition:
                await self.condition.wait_for(lambda: not self.active)
        return self.status()

    @asynccontextmanager
    async def admit(self, session_id: str, disconnected: Callable[[], bool] | None = None,
                    min_version: int = -1):
        request_id = uuid4().hex
        enqueued_at = time.time()
        ticket = None
        async with self.condition:
            if len(self.waiting) >= self.max_waiting:
                raise RuntimeError("turn gate waiting capacity exhausted")
            self.waiting[request_id] = session_id
            try:
                while True:
                    if self.closed or session_id in self.cancelled:
                        raise TurnCancelled(self.cancelled.get(session_id, "shutdown"))
                    if disconnected is not None and disconnected():
                        raise TurnCancelled("client_disconnected_before_admission")
                    busy = any(t.session_id == session_id for t in self.active.values())
                    version_ready = self.version >= min_version
                    if self.accepting and not busy and version_ready and len(self.active) < self.max_active:
                        ticket = TurnTicket(
                            request_id, session_id, uuid4().int % (2**63 - 1),
                            self.version, self.source_optimizer_step, self.epoch,
                            enqueued_at, time.time(),
                        )
                        # No await between checking admission and publishing the ticket.
                        self.active[request_id] = ticket
                        break
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=0.2)
                    except TimeoutError:
                        pass
            finally:
                self.waiting.pop(request_id, None)
        try:
            yield ticket
        finally:
            async with self.condition:
                self.active.pop(request_id, None)
                self.condition.notify_all()
