from unittest.mock import NonCallableMagicMock, call

from mltome.neptune import NeptuneObserver


def test_neptuneobserver():
    ctx_mock = NonCallableMagicMock()
    no = NeptuneObserver(ctx=ctx_mock)

    no.started_event(None, 'train', None, None, {
        'model_id': 'id1',
        'tags': ['1', '2']
    }, None, None)

    ctx_mock.properties.__setitem__.assert_called_with('model_id', 'id1_train')
    ctx_mock.tags.append.assert_has_calls(
        [call('1'), call('2')], any_order=True)

    no.completed_event(None, [1])
    assert not ctx_mock.channel_send.call_count

    no.completed_event(None, [1, 2])

    ctx_mock.channel_send.assert_has_calls(
        [call('valid', 0, 1), call('train', 0, 2)], any_order=True)
