"""Multi-version route ownership without starting a Ray cluster."""
from unittest.mock import patch

import pytest

from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment


def segment(text, version=None):
    ids = [ord(c) for c in text]
    return TokenizedSegment(text=text, token_ids=ids,
                            labels=ids if version is not None else None,
                            logprobs=[-0.1 - (version or 0)] * len(ids),
                            model_step=version, policy_version=version)


def turn(store, prompt, answer, version, request):
    prefix, nodes = store.search("s", prompt, True)
    response = segment(answer, version)
    response.request_id = request
    route = object()
    store.commit_turn("s", prompt, prompt + answer, [n.value for n in nodes],
                      segment(prompt[len(prefix):]) if prefix != prompt else None,
                      response, [route], {"request_id": request, "version": version,
                                          "routing_bytes": len(prompt + answer) * 4})
    return route


def test_complete_last_routes_keep_old_probabilities_and_labels():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    r0 = turn(store, "P", "A", 0, "t0")
    r1 = turn(store, "PAT", "B", 1, "t1")
    data = store.export_training_trace("s", "PATB")
    assert data["labels"] == [-100, ord("A"), -100, ord("B")]
    assert data["logprobs"][1::2] == [-0.1, -1.1]
    assert data["model_steps"] == [None, 0, None, 1]
    assert data["policy_versions"] == [None, 0, None, 1]
    assert data["routed_experts"] == [r1]
    assert data["routing_source_request_id"] == "t1"
    assert store.export_training_trace("s", "PA")["routed_experts"] == [r0]


def test_branch_compact_and_repeated_ancestor_keep_their_own_history():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    turn(store, "P", "A", 0, "t0")
    rb = turn(store, "PAT", "B", 1, "branch")
    rc = turn(store, "PASummary", "C", 2, "compact")
    turn(store, "P", "A", 2, "repeat")
    # Retained descendant still refers to old A even though PA was repeated.
    turn(store, "PATBTool", "D", 2, "next")
    assert store.export_training_trace("s", "PATBToolD")["model_steps"][1] == 0
    assert store.export_training_trace("s", "PA")["model_steps"][1] == 2
    assert store.export_training_trace("s", "PA", request_id="t0")["model_steps"][1] == 0
    with pytest.raises(ValueError, match="identity"):
        store.export_training_trace("s", "PATB", request_id="t0")
    assert store.export_training_trace("s", "PATB")["routed_experts"] == [rb]
    assert store.export_training_trace("s", "PASummaryC")["routed_experts"] == [rc]
    with pytest.raises(ValueError, match="endpoint"):
        store.export_training_trace("s", "PAT")


def test_release_frees_each_request_route_once_and_preserves_other_session():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    r0 = turn(store, "P", "A", 0, "t0")
    r1 = turn(store, "PAT", "B", 1, "t1")
    store.insert("other", "X", segment("X"))
    with patch("ray.internal.free") as free:
        assert store.release_sessions(["s", "s"]) == ["s"]
    assert len(free.call_args_list) == 1
    assert set(free.call_args.args[0]) == {r0, r1}
    assert free.call_args.kwargs == {"local_only": False}
    assert store.list_sessions() == ["other"]
    assert store.turn_memory()["routing_bytes"] == 0


def test_missing_generation_version_rejected():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    bad = segment("A", 0)
    bad.model_step = None
    with pytest.raises(ValueError, match="generation version"):
        store.commit_turn("s", "P", "PA", [], segment("P"), bad, [None],
                          {"request_id": "t0", "version": 0})
    assert store.list_sessions() == []


def test_route_memory_reserves_inflight_before_commit():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    store.turn_byte_limit = 32
    store.reserve_turn("a", 24)
    with pytest.raises(MemoryError):
        store.reserve_turn("b", 16)
    turn(store, "P", "A", 0, "a")
    assert store.turn_memory()["routing_bytes"] == 8
    assert store.turn_memory()["reserved_bytes"] == 0
    store.reserve_turn("b", 24)
    store.release_turn_reservation("b")
    assert store.turn_memory()["reserved_bytes"] == 0


def test_terminal_route_pruning_keeps_both_branches_and_original_probabilities():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    r0 = turn(store, "P", "A", 0, "t0")
    left = turn(store, "PAT", "B", 1, "left")
    right = turn(store, "PAOther", "C", 2, "right")
    with patch("ray.internal.free") as free:
        store.retain_turn_routes("s", ["left", "right"])
        store.retain_turn_routes("s", ["left", "right"])
        free.assert_called_once_with([r0], local_only=False)
    trace = store.export_training_trace("s", "PATB", "left")
    assert trace["routed_experts"] == [left] and trace["model_steps"][1] == 0
    assert store.export_training_trace("s", "PAOtherC", "right")["routed_experts"] == [right]
    with pytest.raises(ValueError, match="released"):
        store.export_training_trace("s", "PA", "t0")
    with patch("ray.internal.free") as free:
        store.release_sessions(["s"])
        assert len(free.call_args_list) == 1
        assert set(free.call_args.args[0]) == {left, right}
        assert free.call_args.kwargs == {"local_only": False}


def test_cross_replica_session_owner_and_cancellation():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    assert store.claim_turn_session('s', 'replica-a')['acquired']
    assert not store.claim_turn_session('s', 'replica-b')['acquired']
    store.release_turn_session('s', 'replica-b')
    assert not store.claim_turn_session('s', 'replica-b')['acquired']
    store.release_turn_session('s', 'replica-a')
    assert store.claim_turn_session('s', 'replica-b')['acquired']
    store.cancel_turn_session('s', 'timeout')
    store.release_turn_session('s', 'replica-b')
    assert store.claim_turn_session('s', 'replica-c')['cancel_reason'] == 'timeout'
