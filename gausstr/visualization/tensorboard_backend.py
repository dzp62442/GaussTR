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
EVAL_TAGS = {'iou', 'miou'}
TIME_TAGS = {'data_time', 'time'}


@VISBACKENDS.register_module()
class GaussTRTensorboardVisBackend(TensorboardVisBackend):
    """只调整 TensorBoard 标量 tag 分组的可视化后端。"""

    @staticmethod
    def _format_scalar_name(name: str) -> str:
        if name in CLASS_IOU_TAGS:
            return f'eval_class/{name}'
        if name in BASIC_TAGS:
            return f'basic/{name}'
        if name in EVAL_TAGS:
            return f'eval/{name}'
        if name in TIME_TAGS:
            return f'time/{name}'
        return name

    def add_scalar(self, name, value, step: int = 0, **kwargs) -> None:
        super().add_scalar(
            self._format_scalar_name(str(name)), value, step, **kwargs)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        assert isinstance(scalar_dict, dict)
        formatted = {
            self._format_scalar_name(str(key)): value
            for key, value in scalar_dict.items()
        }
        super().add_scalars(formatted, step, file_path, **kwargs)
