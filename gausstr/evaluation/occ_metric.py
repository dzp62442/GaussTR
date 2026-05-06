import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.dist import get_dist_info
from terminaltables import AsciiTable

from mmdet3d.evaluation import fast_hist, per_class_iou
from mmdet3d.registry import METRICS


def compute_occ_iou(hist, free_index):
    tp = (
        hist[:free_index, :free_index].sum() +
        hist[free_index + 1:, free_index + 1:].sum())
    return tp / (hist.sum() - hist[free_index, free_index])


@METRICS.register_module()
class OccMetric(BaseMetric):

    def __init__(self,
                 num_classes,
                 use_lidar_mask=False,
                 use_image_mask=True,
                 collect_device='cpu',
                 prefix=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 eval_debug=False,
                 strict_coverage=True,
                 expected_sample_idx_file=None,
                 dump_coverage_prefix=None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super().__init__(
            prefix=prefix, collect_device=collect_device, **kwargs)
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.eval_debug = bool(eval_debug)
        self.strict_coverage = bool(strict_coverage)
        self.dump_coverage_prefix = dump_coverage_prefix
        self.expected_sample_idx = self._load_expected_sample_idx(
            expected_sample_idx_file)

    @staticmethod
    def _load_expected_sample_idx(path):
        if path is None:
            return None
        path = Path(path)
        if path.suffix == '.json':
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'samples' in data:
                return {
                    str(item['sample_idx'])
                    for item in data['samples']
                    if 'sample_idx' in item
                }
            if isinstance(data, dict) and 'by_sample_idx' in data:
                return {str(key) for key in data['by_sample_idx']}
            if isinstance(data, list):
                return {str(item) for item in data}
            raise ValueError(f'Unsupported expected sample json: {path}')
        with path.open('r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}

    @staticmethod
    def _hash_strings(values):
        digest = hashlib.sha256()
        for value in sorted(str(item) for item in values):
            digest.update(value.encode('utf-8'))
            digest.update(b'\0')
        return digest.hexdigest()

    @staticmethod
    def _hash_hist(hist):
        array = np.asarray(hist, dtype=np.int64)
        return hashlib.sha256(array.tobytes()).hexdigest()

    @staticmethod
    def _to_tensor(value, device=None):
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.from_numpy(np.asarray(value))
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @staticmethod
    def _reshape_like(value, reference):
        if value.shape == reference.shape:
            return value
        if value.numel() != reference.numel():
            raise ValueError(
                f'Cannot align shape {tuple(value.shape)} to '
                f'{tuple(reference.shape)}.')
        return value.reshape(reference.shape)

    @staticmethod
    def _sample_idx(data_sample):
        sample_idx = getattr(data_sample, 'sample_idx', None)
        if sample_idx is None and hasattr(data_sample, 'metainfo'):
            sample_idx = data_sample.metainfo.get('sample_idx')
        if sample_idx is None:
            raise KeyError('Data sample missing global sample_idx.')
        return str(sample_idx)

    def process(self, data_batch, data_samples):
        batch_samples = data_batch['data_samples']
        if len(data_samples) != len(batch_samples):
            raise ValueError(
                f'Prediction/data sample count mismatch: '
                f'{len(data_samples)} vs {len(batch_samples)}.')
        rank, _ = get_dist_info()

        for pred, data_sample in zip(data_samples, batch_samples):
            pred = self._to_tensor(pred)
            label = self._to_tensor(
                data_sample.gt_pts_seg.semantic_seg, device=pred.device)
            label = self._reshape_like(label, pred)

            if self.use_image_mask:
                mask = self._to_tensor(
                    data_sample.mask_camera, device=pred.device).to(torch.bool)
            elif self.use_lidar_mask:
                mask = self._to_tensor(
                    data_sample.mask_lidar, device=pred.device).to(torch.bool)
            else:
                mask = None
            if mask is not None:
                mask = self._reshape_like(mask, pred)
                pred = pred[mask]
                label = label[mask]

            pred = pred.flatten().detach().cpu().numpy()
            label = label.flatten().detach().cpu().numpy()
            hist = fast_hist(pred, label, self.num_classes)
            self.results.append({
                'sample_idx': self._sample_idx(data_sample),
                'hist': np.asarray(hist, dtype=np.int64),
                'rank': int(rank),
            })

    def _coverage_summary(self, results, hist):
        sample_ids = [str(item['sample_idx']) for item in results]
        seen = set()
        duplicates = []
        for sample_idx in sample_ids:
            if sample_idx in seen:
                duplicates.append(sample_idx)
            seen.add(sample_idx)

        missing = set()
        unexpected = set()
        if self.expected_sample_idx is not None:
            missing = self.expected_sample_idx - seen
            unexpected = seen - self.expected_sample_idx

        rank_counts = {}
        for item in results:
            rank = str(item.get('rank', 'unknown'))
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        summary = {
            'num_results': len(results),
            'num_unique_samples': len(seen),
            'num_duplicates': len(duplicates),
            'duplicate_preview': sorted(set(duplicates))[:20],
            'num_missing': len(missing),
            'missing_preview': sorted(missing)[:20],
            'num_unexpected': len(unexpected),
            'unexpected_preview': sorted(unexpected)[:20],
            'rank_counts': dict(sorted(rank_counts.items())),
            'sample_idx_sha256': self._hash_strings(seen),
            'hist_sha256': self._hash_hist(hist),
        }
        return summary

    def _validate_coverage(self, summary):
        errors = []
        if summary['num_results'] <= 0:
            errors.append('empty metric results')
        if summary['num_duplicates'] > 0:
            errors.append(
                f'duplicate sample_idx: {summary["duplicate_preview"]}')
        if summary['num_missing'] > 0:
            errors.append(f'missing sample_idx: {summary["missing_preview"]}')
        if summary['num_unexpected'] > 0:
            errors.append(
                f'unexpected sample_idx: {summary["unexpected_preview"]}')
        if errors and self.strict_coverage:
            raise RuntimeError(
                'OccMetric coverage validation failed:\n' +
                '\n'.join(f'- {error}' for error in errors))

    def _dump_coverage(self, summary):
        if self.dump_coverage_prefix is None:
            return
        path = Path(f'{self.dump_coverage_prefix}.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write('\n')

    def _expected_sample_idx(self):
        if self.expected_sample_idx is not None:
            return self.expected_sample_idx
        dataset_meta = getattr(self, 'dataset_meta', None) or {}
        expected = dataset_meta.get('expected_sample_idx')
        if expected is None:
            return None
        return {str(item) for item in expected}

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for item in results:
            hist += np.asarray(item['hist'], dtype=np.int64)

        self.expected_sample_idx = self._expected_sample_idx()
        coverage = self._coverage_summary(results, hist)
        self._validate_coverage(coverage)
        if self.eval_debug:
            print_log(
                'OccMetric coverage: '
                f'num_results={coverage["num_results"]}, '
                f'unique={coverage["num_unique_samples"]}, '
                f'duplicates={coverage["num_duplicates"]}, '
                f'missing={coverage["num_missing"]}, '
                f'unexpected={coverage["num_unexpected"]}, '
                f'sample_idx_sha256={coverage["sample_idx_sha256"]}, '
                f'hist_sha256={coverage["hist_sha256"]}, '
                f'rank_counts={coverage["rank_counts"]}',
                logger=logger)
        self._dump_coverage(coverage)

        iou = per_class_iou(hist)
        # if ignore_index is in iou, replace it with nan
        miou = np.nanmean(iou[:-1])  # NOTE: ignore free class
        label2cat = self.dataset_meta['label2cat']

        header = ['classes']
        for i in range(len(label2cat) - 1):
            header.append(label2cat[i])
        header.extend(['miou', 'iou'])

        ret_dict = dict()
        table_columns = [['results']]
        for i in range(len(label2cat) - 1):
            ret_dict[label2cat[i]] = float(iou[i])
            table_columns.append([f'{iou[i]:.4f}'])
        ret_dict['miou'] = float(miou)
        ret_dict['iou'] = compute_occ_iou(hist, self.num_classes - 1)
        table_columns.append([f'{miou:.4f}'])
        table_columns.append([f"{ret_dict['iou']:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        return ret_dict
