import re
from pathlib import Path
from typing import Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import BaseLoop

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class AutoResumeHook(Hook):
    """Auto resume training from checkpoints under the current work_dir."""

    priority = 'HIGHEST'

    def __init__(self, recursive=True):
        self.recursive = recursive

    def before_run(self, runner):
        if not isinstance(getattr(runner, '_train_loop', None), BaseLoop):
            return
        if getattr(runner, '_resume', False) or getattr(runner, '_load_from',
                                                       None):
            return

        ckpt_path = self._find_latest_checkpoint(
            Path(runner.work_dir), getattr(runner, 'timestamp', None))
        if ckpt_path is None:
            print_log(
                f'AutoResumeHook did not find a checkpoint under '
                f'{runner.work_dir}. Training will start from scratch.',
                logger='current')
            return

        runner._resume = True
        runner._load_from = str(ckpt_path)
        print_log(
            f'AutoResumeHook will resume training from {ckpt_path}.',
            logger='current')

    def _find_latest_checkpoint(self, work_dir: Path,
                                timestamp: Optional[str]) -> Optional[Path]:
        if not work_dir.exists():
            return None

        last_ckpt = self._read_last_checkpoint(work_dir)
        if last_ckpt is not None:
            return last_ckpt

        ckpt_path = self._find_best_checkpoint_in_dir(work_dir)
        if ckpt_path is not None:
            return ckpt_path

        search_dirs = []
        if self.recursive:
            search_dirs = sorted(
                [p for p in work_dir.iterdir() if p.is_dir()],
                key=lambda p: p.name,
                reverse=True)

        for search_dir in search_dirs:
            if timestamp is not None and search_dir.name == timestamp:
                continue

            last_ckpt = self._read_last_checkpoint(search_dir)
            if last_ckpt is not None:
                return last_ckpt

            ckpt_path = self._find_best_checkpoint_in_dir(search_dir)
            if ckpt_path is not None:
                return ckpt_path

        return None

    def _find_best_checkpoint_in_dir(self, directory: Path) -> Optional[Path]:
        candidates = list(directory.glob('epoch_*.pth'))
        candidates.extend(directory.glob('iter_*.pth'))

        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return None

        return max(candidates, key=self._checkpoint_sort_key)

    def _read_last_checkpoint(self, directory: Path) -> Optional[Path]:
        last_checkpoint = directory / 'last_checkpoint'
        if not last_checkpoint.is_file():
            return None

        ckpt = Path(last_checkpoint.read_text().strip())
        if not ckpt.is_absolute():
            ckpt = directory / ckpt
        return ckpt if ckpt.is_file() else None

    def _checkpoint_sort_key(self, path: Path):
        match = re.match(r'^(epoch|iter)_(\d+)\.pth$', path.name)
        step = int(match.group(2)) if match else -1
        kind_rank = 1 if match and match.group(1) == 'epoch' else 0
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        return step, kind_rank, mtime, str(path)
