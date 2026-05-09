import copy

from mmengine.hooks import Hook
from mmengine.logging import MMLogger, print_log

try:
    from mmdet3d.registry import HOOKS, LOOPS
except ImportError:
    from mmdet3d.registry import HOOKS
    from mmengine.registry import LOOPS


@HOOKS.register_module()
class FinalFullEvalHook(Hook):
    """训练全部结束后，额外执行一次完整评估集验证。

    常规训练中的 epoch 后验证仍由 runner 自身的 ``val_dataloader`` 负责。
    本 hook 只在 ``after_train`` 阶段临时构建一套独立的 ValLoop，用于
    跑完整评估集，避免把训练中的 mini 验证和最终完整验证混在一起。
    """

    priority = 'LOW'

    def __init__(self,
                 dataloader,
                 evaluator,
                 loop_cfg=None,
                 enabled=True):
        self.dataloader = copy.deepcopy(dataloader)
        self.evaluator = copy.deepcopy(evaluator)
        self.loop_cfg = copy.deepcopy(loop_cfg or dict(type='ValLoop'))
        self.enabled = bool(enabled)

    def after_train(self, runner) -> None:
        if not self.enabled:
            return

        logger = MMLogger.get_current_instance()
        print_log('开始训练结束后的完整评估。', logger=logger)

        loop_cfg = copy.deepcopy(self.loop_cfg)
        loop_cfg.update(
            runner=runner,
            dataloader=copy.deepcopy(self.dataloader),
            evaluator=copy.deepcopy(self.evaluator))
        loop = LOOPS.build(loop_cfg)
        metrics = loop.run()

        print_log(f'训练结束后的完整评估完成：{metrics}', logger=logger)
