_base_ = './gausstr_featup.py'

log_processor = dict(window_size=50, by_epoch=True)

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromShards',
        _scope_='mmdet3d',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(
        type='ImageAug3D',
        _scope_='mmdet3d',
        final_dim=(432, 768),
        resize_lim=[0.48, 0.48],
        is_train=True),
    dict(
        type='LoadShardedFeatMaps',
        _scope_='mmdet3d',
        group='depth_metric3d',
        key='depth',
        apply_aug=True),
    dict(
        type='LoadShardedFeatMaps',
        _scope_='mmdet3d',
        group='feats_featup',
        key='feats'),
    dict(
        type='LoadShardedFeatMaps',
        _scope_='mmdet3d',
        group='sem_seg_grounded_sam2',
        key='sem_seg',
        apply_aug=True),
    dict(
        type='Pack3DDetInputs',
        _scope_='mmdet3d',
        keys=['img'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats', 'sem_seg'
        ])
]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromShards',
        _scope_='mmdet3d',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(type='LoadShardedOccFromArrays', _scope_='mmdet3d', group='occ_gt'),
    dict(
        type='ImageAug3D',
        _scope_='mmdet3d',
        final_dim=(432, 768),
        resize_lim=[0.48, 0.48]),
    dict(
        type='LoadShardedFeatMaps',
        _scope_='mmdet3d',
        group='depth_metric3d',
        key='depth',
        apply_aug=True),
    dict(
        type='LoadShardedFeatMaps',
        _scope_='mmdet3d',
        group='feats_featup',
        key='feats'),
    dict(
        type='Pack3DDetInputs',
        _scope_='mmdet3d',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats', 'mask_camera'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    sampler=dict(
        type='ShardAwareSampler',
        shuffle=True,
        num_workers=1,
        prefetch_shards=0,
        sample_shuffle_block_size=16,
        prefetch_samples=16),
    dataset=dict(
        _delete_=True,
        type='NuScenesOccShardedDataset',
        shard_root='data/gausstr_shards',
        split='train',
        preload_mode='lazy',
        require_success=False,
        max_cache_bytes=24 * 1024**3,
        prefetch_shards=0,
        prefetch_workers=0,
        prefetch_max_tasks_per_call=0,
        debug=False,
        debug_interval=100,
        slow_log_threshold=1.0,
        serialize_data=False,
        required_groups=dict(
            raw='raw_nuscenes',
            depth='depth_metric3d',
            feats='feats_featup',
            sem_seg='sem_seg_grounded_sam2'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='NuScenesOccShardedDataset',
        shard_root='data/gausstr_shards',
        split='val',
        preload_mode='lazy',
        require_success=False,
        max_cache_bytes=24 * 1024**3,
        prefetch_shards=0,
        prefetch_workers=0,
        prefetch_max_tasks_per_call=0,
        debug=False,
        debug_interval=100,
        slow_log_threshold=1.0,
        serialize_data=False,
        required_groups=dict(
            raw='raw_nuscenes',
            depth='depth_metric3d',
            feats='feats_featup',
            occ='occ_gt'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader
