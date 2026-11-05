import asyncio
import unittest

from xtuner.v1.rl.rollout.turn_gate import TurnCancelled, TurnGate


class TestTurnGate(unittest.IsolatedAsyncioTestCase):
    async def publish(self, gate, version, epoch):
        await gate.prepare_version(version, version, epoch)
        await gate.open_admission(version, version, epoch)

    async def test_drain_preserves_old_turn_and_versions_queued_turn(self):
        gate = TurnGate()
        await self.publish(gate, 0, 0)
        started, finish, queued_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        versions = []

        async def old():
            async with gate.admit("a") as ticket:
                started.set()
                await finish.wait()
                ticket.phase = "committing"
                versions.append(ticket.version)
                await asyncio.sleep(0)
                ticket.trace_committed = True

        async def queued():
            async with gate.admit("b") as ticket:
                versions.append(ticket.version)
                queued_done.set()

        first = asyncio.create_task(old())
        await started.wait()
        await gate.close_admission(1)
        second = asyncio.create_task(queued())
        with self.assertRaises(TimeoutError):
            await gate.drain(0.01)
        with self.assertRaises(RuntimeError):
            await gate.prepare_version(1, 1, 1)
        self.assertFalse(queued_done.is_set())
        finish.set()
        await first
        await gate.drain(1)
        self.assertFalse(queued_done.is_set())
        await self.publish(gate, 1, 1)
        await second
        self.assertEqual(versions, [0, 1])

    async def test_cancel_only_target_and_keeps_active_until_exit(self):
        gate = TurnGate()
        await self.publish(gate, 0, 0)
        async with gate.admit("target") as ticket:
            async with gate.admit("healthy") as healthy:
                cancelled = await gate.cancel_session("target", "drain_timeout")
                self.assertEqual(cancelled[0]["engine_session_id"], ticket.engine_session_id)
                self.assertIsNone(healthy.cancel_reason)
                self.assertEqual(gate.status()["active"], 2)
                self.assertEqual(ticket.cancel_reason, "drain_timeout")
        with self.assertRaises(TurnCancelled):
            async with gate.admit("target"):
                self.fail("cancelled session re-admitted")

    async def test_cancel_waiter_and_disconnect_do_not_reach_engine(self):
        gate = TurnGate()
        async def request():
            async with gate.admit("waiting"):
                self.fail("closed gate forwarded a request")
        task = asyncio.create_task(request())
        await asyncio.sleep(0)
        await gate.cancel_session("waiting", "cancelled")
        with self.assertRaises(TurnCancelled):
            await task
        with self.assertRaises(TurnCancelled):
            async with gate.admit("gone", lambda: True):
                self.fail("disconnected request admitted")
        self.assertEqual(gate.status()["waiting"], 0)

    async def test_session_serialization_shutdown_and_publication_guards(self):
        gate = TurnGate()
        await self.publish(gate, 0, 0)
        entered = asyncio.Event()
        async def another():
            async with gate.admit("same"):
                entered.set()
        async with gate.admit("same"):
            task = asyncio.create_task(another())
            await asyncio.sleep(0.01)
            self.assertFalse(entered.is_set())
            await gate.shutdown()
        with self.assertRaises(TurnCancelled):
            await task
        with self.assertRaises(RuntimeError):
            await self.publish(gate, 1, 1)
