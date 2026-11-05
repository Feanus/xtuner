import asyncio
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout.session_server import SessionServer, _turn_context
from xtuner.v1.rl.rollout.turn_gate import TurnGate


class TestTurnHTTP(unittest.IsolatedAsyncioTestCase):
    async def test_pause_waits_for_commit_then_body_carries_admitted_version(self):
        server = SessionServer.__new__(SessionServer)
        server.turn_gate = TurnGate()
        server.store = NS(release_turn_reservation=NS(remote=AsyncMock()),
                          claim_turn_session=NS(remote=AsyncMock(return_value={'acquired': True, 'cancel_reason': None})),
                          release_turn_session=NS(remote=AsyncMock()),
                          smoke_required_version=NS(remote=AsyncMock(return_value=-1)))
        server._abort_turn = AsyncMock()
        started, commit = asyncio.Event(), asyncio.Event()
        async def core(request):
            ticket = _turn_context.get()['ticket']
            started.set()
            await commit.wait()
            ticket.trace_committed = True
            return web.json_response({'id': 'original', 'choices': []})
        server._handle_request_core = core
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', server._handle_request)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            await server.turn_gate.prepare_version(0, 0, 0)
            await server.turn_gate.open_admission(0, 0, 0)
            response_task = asyncio.create_task(client.post('/v1/chat/completions', json={
                'session_id': 's', 'messages': [{'role': 'user', 'content': 'hello'}]}))
            await asyncio.wait_for(started.wait(), 2)
            await server.turn_gate.close_admission(1)
            with self.assertRaises(TimeoutError):
                await server.turn_gate.drain(0.01)
            commit.set()
            response = await response_task
            body = await response.json()
            self.assertEqual(response.status, 200)
            self.assertEqual(body['id'], 'xtuner-' + response.headers['X-XTuner-Request-ID'])
            self.assertEqual(response.headers['X-XTuner-Model-Step'], '0')
            await server.turn_gate.drain(1)
            await server.turn_gate.prepare_version(1, 1, 1)
            await server.turn_gate.open_admission(1, 1, 1)
            server._abort_turn.assert_not_awaited()
        finally:
            await client.close()

    async def test_recorder_ack_reaches_store_before_generation_and_is_not_forwarded(self):
        from lagent.adapters.proxy import SessionClient
        from unittest.mock import patch
        server = SessionServer.__new__(SessionServer)
        server.turn_gate = TurnGate()
        ack = AsyncMock()
        server.store = NS(release_turn_reservation=NS(remote=AsyncMock()),
                          claim_turn_session=NS(remote=AsyncMock(return_value={'acquired': True, 'cancel_reason': None})),
                          release_turn_session=NS(remote=AsyncMock()),
                          acknowledge_turn_routes=NS(remote=ack))
        server._abort_turn = AsyncMock()
        bodies = []
        async def core(request):
            body = await request.json()
            bodies.append(body)
            if len(bodies) > 1:
                self.assertEqual(ack.await_count, len(bodies)-1)
            ticket = _turn_context.get()['ticket']
            ticket.trace_committed = True
            return web.json_response({'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'ok'}}]})
        server._handle_request_core = core
        await server.turn_gate.prepare_version(0, 0, 0)
        await server.turn_gate.open_admission(0, 0, 0)
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', server._handle_request)
        upstream = TestServer(app)
        await upstream.start_server()
        proxy = SessionClient(real_api_key='EMPTY', real_base_url=str(upstream.make_url('/v1')), port=0, session_id='s')
        await proxy.start()
        import aiohttp
        try:
            messages = [{'role': 'user', 'content': 'hello'}]
            async with aiohttp.ClientSession() as http:
                for i in range(3):
                    async with http.post(proxy.openai_base_url + '/chat/completions',
                                         json={'messages': messages}) as response:
                        self.assertEqual(response.status, 200)
                        result = await response.json()
                    messages += [result['choices'][0]['message'], {'role': 'user', 'content': 'next'}]
            self.assertNotIn('xtuner_route_ack', bodies[0])
            self.assertEqual(len(bodies[2]['xtuner_route_ack']['observed']), 2)
            self.assertEqual(len(bodies[2]['xtuner_route_ack']['retained']), 1)
            self.assertEqual(ack.call_args.args, ('s', bodies[2]['xtuner_route_ack']))
            from xtuner.v1.rl.rollout.session_server import _SESSION_SERVER_ONLY_KEYS
            self.assertIn('xtuner_route_ack', _SESSION_SERVER_ONLY_KEYS)
        finally:
            await proxy.stop()
            await upstream.close()

    async def test_replaced_entrypoint_rejected_before_open(self):
        from xtuner.v1.rl.rollout.controller import RolloutController
        from unittest.mock import Mock
        controller = RolloutController.__new__(RolloutController)
        remote = AsyncMock(return_value={'instance_id': 'new'})
        workers = [NS(actor=NS(turn_control=NS(remote=remote)))]
        controller.registry = NS(active_workers=lambda: workers, all_workers=lambda: workers)
        controller.health_manager = NS(resume=Mock())
        controller._turn_instances = {'old'}
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            await controller.turn_control('open_admission', version=1, source_optimizer_step=1, epoch=1)
        remote.assert_awaited_once_with('status')
        controller.health_manager.resume.assert_not_called()

    async def test_same_session_on_two_http_entrypoints_waits_for_new_publication(self):
        from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore
        store = RolloutTraceStore.__ray_metadata__.modified_class()
        shared = NS(**{name: NS(remote=AsyncMock(side_effect=getattr(store, name))) for name in (
            'claim_turn_session', 'release_turn_session', 'release_turn_reservation', 'smoke_required_version')})
        first_started, finish_first, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        versions = []
        clients, servers = [], []
        def handler(index):
            async def core(request):
                ticket = _turn_context.get()['ticket']
                versions.append(ticket.version)
                if index == 0:
                    first_started.set()
                    await finish_first.wait()
                else:
                    second_started.set()
                ticket.trace_committed = True
                return web.json_response({'id': 'initial', 'choices': []})
            return core
        tasks = []
        try:
            for index in range(2):
                server = SessionServer.__new__(SessionServer)
                server.turn_gate, server.store = TurnGate(), shared
                server._abort_turn = AsyncMock()
                server._handle_request_core = handler(index)
                await server.turn_gate.prepare_version(0, 0, 0)
                await server.turn_gate.open_admission(0, 0, 0)
                app = web.Application()
                app.router.add_route('*', '/{path:.*}', server._handle_request)
                client = TestClient(TestServer(app))
                await client.start_server()
                servers.append(server)
                clients.append(client)
            payload = {'session_id': 'shared', 'messages': [{'role': 'user', 'content': 'hello'}]}
            tasks.append(asyncio.create_task(clients[0].post('/v1/chat/completions', json=payload)))
            await asyncio.wait_for(first_started.wait(), 2)
            tasks.append(asyncio.create_task(clients[1].post('/v1/chat/completions', json=payload)))
            for _ in range(100):
                if shared.claim_turn_session.remote.await_count >= 2:
                    break
                await asyncio.sleep(0.01)
            self.assertGreaterEqual(shared.claim_turn_session.remote.await_count, 2)
            self.assertFalse(second_started.is_set())
            for server in servers:
                await server.turn_gate.close_admission(1)
            finish_first.set()
            first = await asyncio.wait_for(tasks[0], 2)
            self.assertEqual(first.status, 200)
            for server in servers:
                await server.turn_gate.drain(2)
            self.assertFalse(second_started.is_set())
            for server in servers:
                await server.turn_gate.prepare_version(1, 1, 1)
                await server.turn_gate.open_admission(1, 1, 1)
            second = await asyncio.wait_for(tasks[1], 2)
            self.assertEqual(second.status, 200)
            self.assertEqual(versions, [0, 1])
            self.assertEqual(store.turn_owners, {})
        finally:
            finish_first.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for client in clients:
                await client.close()
