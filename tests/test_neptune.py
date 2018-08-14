from unittest.mock import NonCallableMagicMock, call

from mltome.neptune import NeptuneObserver, NeptuneSkorchCallback


def test_neptuneobserver():
    ctx_mock = NonCallableMagicMock()
    no = NeptuneObserver(ctx_mock, 'model_id', 'tags')

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


def test_neptunecallback():
    ctx_mock = NonCallableMagicMock()
    no = NeptuneSkorchCallback(
        ctx_mock, batch_targets=['train_loss'], epoch_targets=['valid_loss'])

    no.update_batch_values({'train_loss': 1}, 0)
    ctx_mock.channel_send.assert_has_calls([call('train_loss', 0, 1)])

    no.update_epoch_values({'valid_loss': 1}, 0)
    ctx_mock.channel_send.assert_has_calls([call('valid_loss', 0, 1)])
