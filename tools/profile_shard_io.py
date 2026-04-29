#!/usr/bin/env python
"""Profile shard file read bandwidth and torch.load latency.

This script is intentionally independent from the training runner so TOS/FUSE
read performance can be checked without initializing models or dataloaders.
"""

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-root', default='data/gausstr_shards')
    parser.add_argument('--split', default='train')
    parser.add_argument(
        '--groups',
        nargs='+',
        default=[
            'raw_nuscenes',
            'depth_metric3d',
            'feats_featup',
            'sem_seg_grounded_sam2',
        ])
    parser.add_argument('--num-files', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--mode',
        choices=['read', 'torch_load', 'both'],
        default='both')
    parser.add_argument('--read-chunk-mb', type=int, default=16)
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='Number of concurrent file loads per group.')
    parser.add_argument('--no-random', action='store_true')
    return parser.parse_args()


def load_manifest(split_root: Path, group: str):
    manifest_path = split_root / group / 'group_manifest.json'
    with manifest_path.open('r', encoding='utf-8') as f:
        manifest = json.load(f)
    files = []
    for shard_id, entry in sorted(manifest['shards'].items()):
        files.append((shard_id, split_root / entry['path'], int(entry['bytes'])))
    return files


def read_file(path: Path, chunk_size: int) -> int:
    total = 0
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
    return total


def torch_load_file(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def time_call(fn, *args):
    started = time.monotonic()
    result = fn(*args)
    return time.monotonic() - started, result


def profile_one(mode: str, shard_id: str, path: Path, size: int,
                chunk_size: int):
    if mode == 'read':
        elapsed, read_bytes = time_call(read_file, path, chunk_size)
        measured_size = int(read_bytes)
        samples = None
    else:
        elapsed, payload = time_call(torch_load_file, path)
        measured_size = size
        samples = len(payload.get('samples', []))
        del payload
    mbps = measured_size / 1024**2 / max(elapsed, 1e-9)
    return dict(
        shard_id=shard_id,
        path=path,
        bytes=measured_size,
        elapsed=elapsed,
        mbps=mbps,
        samples=samples)


def fmt_mib(num_bytes: int) -> str:
    return f'{num_bytes / 1024**2:.2f}'


def print_summary(group: str, mode: str, rows):
    if not rows:
        return
    elapsed = [row['elapsed'] for row in rows]
    mbps = [row['mbps'] for row in rows]
    print(
        f'SUMMARY group={group} mode={mode} '
        f'n={len(rows)} '
        f'avg_s={statistics.mean(elapsed):.2f} '
        f'p50_s={statistics.median(elapsed):.2f} '
        f'max_s={max(elapsed):.2f} '
        f'avg_mib_s={statistics.mean(mbps):.2f} '
        f'min_mib_s={min(mbps):.2f}',
        flush=True)


def print_load(group: str, mode: str, row):
    samples = '' if row['samples'] is None else f' samples={row["samples"]}'
    print(
        f'LOAD group={group} shard={row["shard_id"]} mode={mode} '
        f'bytes_mib={fmt_mib(row["bytes"])} '
        f'elapsed_s={row["elapsed"]:.2f} mib_s={row["mbps"]:.2f}'
        f'{samples} path={row["path"]}',
        flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    split_root = Path(args.shard_root) / args.split
    chunk_size = args.read_chunk_mb * 1024**2

    print(
        f'PROFILE shard_root={args.shard_root} split={args.split} '
        f'mode={args.mode} num_files={args.num_files} '
        f'concurrency={args.concurrency} pid={os.getpid()}',
        flush=True)

    for group in args.groups:
        files = load_manifest(split_root, group)
        if args.no_random:
            selected = files[:args.num_files]
        else:
            selected = random.sample(files, min(args.num_files, len(files)))
        print(f'GROUP {group} selected={len(selected)}/{len(files)}', flush=True)

        for mode in ('read', 'torch_load'):
            if args.mode not in {mode, 'both'}:
                continue
            rows = []
            started = time.monotonic()
            if args.concurrency <= 1:
                for shard_id, path, size in selected:
                    row = profile_one(mode, shard_id, path, size, chunk_size)
                    rows.append(row)
                    print_load(group, mode, row)
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = [
                        executor.submit(profile_one, mode, shard_id, path, size,
                                        chunk_size)
                        for shard_id, path, size in selected
                    ]
                    for future in as_completed(futures):
                        row = future.result()
                        rows.append(row)
                        print_load(group, mode, row)
            wall_elapsed = time.monotonic() - started
            print_summary(group, mode, rows)
            wall_mib_s = (
                sum(row['bytes'] for row in rows) / 1024**2 /
                max(wall_elapsed, 1e-9))
            print(
                f'WALL group={group} mode={mode} n={len(rows)} '
                f'concurrency={args.concurrency} '
                f'elapsed_s={wall_elapsed:.2f} aggregate_mib_s={wall_mib_s:.2f}',
                flush=True)


if __name__ == '__main__':
    main()
