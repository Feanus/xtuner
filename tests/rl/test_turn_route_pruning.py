"""CPU contracts for recorder-acknowledged live routing reclamation."""
from unittest.mock import patch

import pytest

from lagent.adapters.proxy import SessionClient
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment


def commit(store, prompt, answer, request, version=0):
    prefix, nodes = store.search('s', prompt, True)
    def segment(text, generated=False):
        return TokenizedSegment(text=text, token_ids=list(text.encode()),
            labels=list(text.encode()) if generated else None,
            logprobs=[-.1-version]*len(text), model_step=version if generated else None,
            policy_version=version if generated else None, request_id=request if generated else None)
    route = object()
    store.reserve_turn(request, len(prompt+answer)*4)
    store.commit_turn('s', prompt, prompt+answer, [n.value for n in nodes],
                      segment(prompt[len(prefix):]) if prefix != prompt else None,
                      segment(answer, True), [route],
                      {'request_id':request, 'version':version, 'routing_bytes':len(prompt+answer)*4})
    return route


def record(client, request, messages, tools=None):
    client._records['s'].append({'request_id':request, 'messages':messages, 'tools':tools})


def test_linear_chain_memory_bounded_and_original_probabilities_survive():
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    store.turn_byte_limit = 1000
    client = SessionClient(real_api_key='EMPTY', real_base_url='http://unused', session_id='s')
    messages=[]
    prompt='P'
    with patch('ray.internal.free') as free:
        for i in range(40):
            request=f't{i}'
            route=commit(store,prompt,'A',request,i%3)
            messages += [{'role':'user','content':str(i)}, {'role':'assistant','content':'A'}]
            record(client,request,list(messages))
            store.acknowledge_turn_routes('s',client._route_ack())
            assert store.turn_memory()['routing_bytes']==len(prompt+'A')*4
            assert store.turn_memory()['reserved_bytes']==0
            prompt+='AT'
        assert free.call_count==39
        data=store.export_training_trace('s',prompt[:-1],'t39')
        assert data['routed_experts']==[route]
        assert data['model_steps'][1::2]==[i%3 for i in range(40)]
        assert data['logprobs'][1::2]==[-.1-i%3 for i in range(40)]
        assert sum(x!=-100 for x in data['labels'])==40
        with pytest.raises(ValueError,match='released'):
            store.export_training_trace('s','PA','t0')
        store.release_sessions(['s'])
        assert free.call_count==40
        assert store.turn_memory()['routing_bytes']==0


def test_branches_tools_duplicates_and_unreceived_response():
    store=RolloutTraceStore.__ray_metadata__.modified_class()
    client=SessionClient(real_api_key='EMPTY', real_base_url='http://unused', session_id='s')
    a=[{'role':'user','content':'P'},{'role':'assistant','content':'A'}]
    left=a+[{'role':'user','content':'L'},{'role':'assistant','content':'B'}]
    right=a+[{'role':'user','content':'R'},{'role':'assistant','content':'C'}]
    r0=commit(store,'P','A','t0');record(client,'t0',a)
    rl=commit(store,'PAL','B','left',1);record(client,'left',left)
    rr=commit(store,'PAR','C','right',2);record(client,'right',right)
    # Committed response that never reached the recorder must not be freed.
    unreceived=commit(store,'PALBTool','D','unreceived',2)
    with patch('ray.internal.free') as free:
        store.acknowledge_turn_routes('s',client._route_ack())
        free.assert_called_once_with([r0],local_only=False)
        # Same messages with changed tools retain a separate endpoint.
        tools=commit(store,'OTHER','E','tools',2);record(client,'tools',left,[{'name':'new'}])
        newer=commit(store,'PAL','B','repeat',2);record(client,'repeat',left)
        store.acknowledge_turn_routes('s',client._route_ack())
        assert free.call_count==2
        free.assert_called_with([rl],local_only=False)
        for key,request,route in [('PARC','right',rr),('PALBToolD','unreceived',unreceived),
                                  ('OTHERE','tools',tools),('PALB','repeat',newer)]:
            assert store.export_training_trace('s',key,request)['routed_experts']==[route]


def test_invalid_ack_is_atomic_export_is_pinned_and_retry_is_idempotent():
    store=RolloutTraceStore.__ray_metadata__.modified_class()
    r0=commit(store,'P','A','t0');r1=commit(store,'PAT','B','t1',1)
    before=store.turn_memory()
    with patch('ray.internal.free') as free:
        for ack in [None,{}, {'observed':['t0'],'retained':[]},
                    {'observed':['missing'],'retained':['missing']},
                    {'observed':['t0'],'retained':['t1']},
                    {'observed':['key:PA'],'retained':['key:PA']}]:
            with pytest.raises(ValueError):store.acknowledge_turn_routes('s',ack)
        assert store.turn_memory()==before
        free.assert_not_called()
        # Already exported routes have a consumer and are pinned.
        assert store.export_training_trace('s','PA','t0')['routed_experts']==[r0]
        ack={'observed':['t0','t1'],'retained':['t1']}
        store.acknowledge_turn_routes('s',ack)
        store.acknowledge_turn_routes('s',ack)
        free.assert_not_called()
        # A failed commit cannot replace a valid endpoint.
        bad=TokenizedSegment(text='C',token_ids=[67],labels=[67],logprobs=[-.1])
        with pytest.raises(ValueError):store.commit_turn('s','PATB','PATBC',[],None,bad,[object()],{'request_id':'bad'})
        assert store.turn_memory()==before
        assert store.export_training_trace('s','PATB','t1')['routed_experts']==[r1]


def test_stale_ack_cannot_resurrect_routes_or_release_a_new_branch():
    store=RolloutTraceStore.__ray_metadata__.modified_class()
    r0=commit(store,'P','A','t0');r1=commit(store,'PAT','B','t1',1)
    with patch('ray.internal.free') as free:
        ack={'observed':['t0','t1'],'retained':['t1']}
        store.acknowledge_turn_routes('s',ack)
        store.acknowledge_turn_routes('s',ack)
        free.assert_called_once_with([r0],local_only=False)
        branch=commit(store,'PAOther','C','branch',2)
        before=store.turn_memory()
        assert store.acknowledge_turn_routes('s',{'observed':['t0'],'retained':['t0']}) == {
            'released_routes':0,'released_bytes':0}
        assert store.turn_memory()==before
        free.assert_called_once()
        assert store.export_training_trace('s','PAOtherC','branch')['routed_experts']==[branch]
        assert store.export_training_trace('s','PATB','t1')['routed_experts']==[r1]


def test_recorder_missing_identity_disables_ack():
    client=SessionClient(real_api_key='EMPTY',real_base_url='http://unused',session_id='s')
    record(client,None,[{'role':'assistant','content':'old'}])
    record(client,'new',[{'role':'assistant','content':'new'}])
    assert client._route_ack() is None
