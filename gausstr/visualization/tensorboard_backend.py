import os
from pathlib import Path
from typing import Optional

from mmengine.visualization import TensorboardVisBackend

try:
    from mmdet3d.registry import VISBACKENDS
except ImportError:
    from mmengine.registry import VISBACKENDS


CLASS_IOU_TAGS = {
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
}

BASIC_TAGS = {'epoch', 'iter', 'lr', 'memory'}
EVAL_TAGS = {'iou', 'miou*', 'miou'}
TIME_TAGS = {'data_time', 'time'}


@VISBACKENDS.register_module()
class GaussTRTensorboardVisBackend(TensorboardVisBackend):
    """只调整 TensorBoard 标量 tag 分组的可视化后端。"""

    def __init__(self,
                 save_dir: str,
                 log_dir_name: str = 'tf',
                 train_only: bool = True,
                 purge_resume: bool = True):
        super().__init__(save_dir)
        self._log_dir_name = log_dir_name
        self._train_only = train_only
        self._purge_resume = purge_resume
        self._enabled = not train_only
        self._purge_step = 0
        self._runner_work_dir = None
        self._current_iter = None

    def configure_for_runner(self, runner) -> None:
        """Enable TensorBoard for training and bind it to the experiment dir."""
        self._enabled = True
        self._runner_work_dir = str(runner.work_dir)
        self._purge_step = int(getattr(runner, 'iter', 0) or 0)
        self._current_iter = self._purge_step

    def set_current_iter(self, runner) -> None:
        self._current_iter = int(getattr(runner, 'iter', 0) or 0)

    def _resolve_log_dir(self) -> str:
        if self._runner_work_dir is not None:
            work_dir = Path(self._runner_work_dir)
            return str(work_dir / self._log_dir_name
                       if self._log_dir_name else work_dir)

        save_dir = Path(self._save_dir)
        if save_dir.name == 'vis_data' and save_dir.parent.parent != save_dir:
            exp_dir = save_dir.parent.parent
            return str(exp_dir / self._log_dir_name
                       if self._log_dir_name else exp_dir)
        return str(save_dir / self._log_dir_name
                   if self._log_dir_name else save_dir)

    def _init_env(self):
        if self._train_only and not self._enabled:
            return
        self._save_dir = self._resolve_log_dir()
        os.makedirs(self._save_dir, exist_ok=True)
        if self._purge_resume:
            self._tensorboard = self._build_writer(
                self._save_dir, purge_step=self._purge_step)
        else:
            self._tensorboard = self._build_writer(self._save_dir)

    @staticmethod
    def _build_writer(log_dir: str, purge_step: Optional[int] = None):
        if purge_step is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                return SummaryWriter(log_dir=log_dir, purge_step=purge_step)
            except ImportError:
                from tensorboardX import SummaryWriter
                return SummaryWriter(logdir=log_dir, purge_step=purge_step)
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except ImportError:
            from tensorboardX import SummaryWriter
            return SummaryWriter(logdir=log_dir)

    def add_config(self, config, **kwargs) -> None:
        if self._train_only and not self._enabled:
            return
        super().add_config(config, **kwargs)

    def add_graph(self, model, data_batch, **kwargs) -> None:
        if self._train_only and not self._enabled:
            return
        super().add_graph(model, data_batch, **kwargs)

    def add_image(self, name, image, step: int = 0, **kwargs) -> None:
        if self._train_only and not self._enabled:
            return
        super().add_image(name, image, step, **kwargs)

    @staticmethod
    def _format_scalar_name(name: str) -> str:
        if name in CLASS_IOU_TAGS:
            return f'eval_classes/{name}'
        if name in BASIC_TAGS:
            return f'basic/{name}'
        if name in EVAL_TAGS:
            return f'eval_all/{name}'
        if name in TIME_TAGS:
            return f'time/{name}'
        return name

    @staticmethod
    def _contains_eval_scalar(scalar_dict: dict) -> bool:
        eval_names = CLASS_IOU_TAGS | EVAL_TAGS
        return any(str(key) in eval_names for key in scalar_dict)

    def add_scalar(self, name, value, step: int = 0, **kwargs) -> None:
        if self._train_only and not self._enabled:
            return
        super().add_scalar(
            self._format_scalar_name(str(name)), value, step, **kwargs)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        if self._train_only and not self._enabled:
            return
        assert isinstance(scalar_dict, dict)
        if (self._contains_eval_scalar(scalar_dict)
                and self._current_iter is not None):
            step = self._current_iter
        formatted = {
            self._format_scalar_name(str(key)): value
            for key, value in scalar_dict.items()
        }
        super().add_scalars(formatted, step, file_path, **kwargs)
