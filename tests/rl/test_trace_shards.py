import asyncio
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from xtuner.v1.rl.rollout.trace_shards import ShardedTraceStore, shard_count


def test_stable_direct_session_routing_and_reservation():
    actors=[Mock() for _ in range(4)]
    store=ShardedTraceStore(actors)
    for sid in ['s','long-session','1234']:
        index=store.index(sid)
        assert index==ShardedTraceStore(actors).index(sid)
        store.prepare_turn.remote(sid,'P','r')
        actors[index].prepare_turn.remote.assert_called_with(sid,'P','r')
        store.reserve_turn.remote('r',10,session_id=sid)
        actors[index].reserve_turn.remote.assert_called_with('r',10,session_id=sid)
        store.release_turn_reservation.remote('r',session_id=sid)
        actors[index].release_turn_reservation.remote.assert_called_with('r',session_id=sid)
    with pytest.raises(ValueError,match='session_id'):store.reserve_turn.remote('r',10)


def test_invalid_shard_count(monkeypatch):
    for n in ('0','65','-1'):
        monkeypatch.setenv('XTUNER_TRACE_SHARDS',n)
        with pytest.raises(ValueError):shard_count()


def test_grouped_cleanup_and_aggregation_return_ray_refs(monkeypatch):
    import xtuner.v1.rl.rollout.trace_shards as module
    actors=[Mock() for _ in range(4)];store=ShardedTraceStore(actors)
    aggregate=Mock(return_value='aggregate-ref')
    monkeypatch.setattr(module,'_collect',SimpleNamespace(remote=aggregate))
    ids=['a','b','a','c']
    assert store.begin_release_sessions.remote(ids)=='aggregate-ref'
    assert aggregate.call_args.args[0]=='begin_release_sessions'
    seen=[]
    for actor in actors:
        if actor.begin_release_sessions.remote.called:
            seen.extend(actor.begin_release_sessions.remote.call_args.args[0])
    assert set(seen)=={'a','b','c'} and len(seen)==3
    assert store.turn_memory.remote()=='aggregate-ref'
    assert len(aggregate.call_args.args[1:])==4
