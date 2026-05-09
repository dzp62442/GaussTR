from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class TensorboardRunHook(Hook):
    """Configure GaussTR TensorBoard backends for a single resumable run."""

    priority = 'VERY_HIGH'

    def _iter_tensorboard_backends(self, runner):
        backends = getattr(runner.visualizer, '_vis_backends', {})
        for backend in backends.values():
            if hasattr(backend, 'configure_for_runner'):
                yield backend

    def before_train(self, runner) -> None:
        for backend in self._iter_tensorboard_backends(runner):
            backend.configure_for_runner(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        for backend in self._iter_tensorboard_backends(runner):
            backend.set_current_iter(runner)

    def before_val_epoch(self, runner) -> None:
        if getattr(runner, '_train_loop', None) is None:
            return
        for backend in self._iter_tensorboard_backends(runner):
            backend.set_current_iter(runner)
