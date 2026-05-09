from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


def _set_dataset_epoch(dataset, epoch: int) -> bool:
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(epoch)
        return True
    updated = False
    for attr in ('dataset', 'datasets'):
        child = getattr(dataset, attr, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for item in child:
                updated = _set_dataset_epoch(item, epoch) or updated
        else:
            updated = _set_dataset_epoch(child, epoch) or updated
    return updated


@HOOKS.register_module()
class ChunkDatasetEpochHook(Hook):
    """Propagate MMEngine epoch to chunk IterableDataset workers."""

    priority = 'NORMAL'

    def before_train_epoch(self, runner) -> None:
        dataset = getattr(runner.train_dataloader, 'dataset', None)
        if dataset is not None:
            _set_dataset_epoch(dataset, runner.epoch)
