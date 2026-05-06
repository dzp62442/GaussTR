_base_ = './gausstr_talk2dino.py'

log_processor = dict(window_size=50, by_epoch=True)

custom_hooks = [
    dict(type='AutoResumeHook'),
    dict(type='ChunkDatasetEpochHook'),
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromChunks',
        _scope_='mmdet3d',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(
        type='ImageAug3D',
        _scope_='mmdet3d',
        final_dim=(504, 896),
        resize_lim=[0.56, 0.56],
        is_train=True),
    dict(
        type='LoadChunkFeatMaps',
        _scope_='mmdet3d',
        key='depth',
        apply_aug=True),
    dict(
        type='Pack3DDetInputs',
        _scope_='mmdet3d',
        keys=['img'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats'
        ])
]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromChunks',
        _scope_='mmdet3d',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(type='LoadChunkOccFromArrays', _scope_='mmdet3d'),
    dict(
        type='ImageAug3D',
        _scope_='mmdet3d',
        final_dim=(504, 896),
        resize_lim=[0.56, 0.56]),
    dict(
        type='LoadChunkFeatMaps',
        _scope_='mmdet3d',
        key='depth',
        apply_aug=True),
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
    sampler=None,
    dataset=dict(
        _delete_=True,
        type='NuScenesOccChunkDataset',
        chunk_root='data/gausstr_chunks',
        split='train',
        profile='talk2dino_metric3d',
        chunk_shuffle=True,
        sample_shuffle=True,
        seed=2026,
        mini=False,
        mini_stride=10,
        mini_offset=0,
        pad_train_chunks=True,
        skip_padding=False,
        load_to_memory=True,
        prefetch_chunks=4,
        prefetch_workers=1,
        cache_chunks_in_memory=False,
        stable_rank_partition=False,
        read_chunk_bytes=16 * 1024 * 1024,
        debug=False,
        slow_log_threshold=1.0,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=1,
    drop_last=False,
    sampler=None,
    dataset=dict(
        _delete_=True,
        type='NuScenesOccChunkDataset',
        chunk_root='data/gausstr_chunks',
        split='val',
        profile='talk2dino_metric3d',
        chunk_shuffle=False,
        sample_shuffle=False,
        seed=2026,
        mini=False,
        mini_stride=10,
        mini_offset=0,
        pad_train_chunks=False,
        skip_padding=True,
        debug=False,
        slow_log_threshold=1.0,
        pipeline=val_pipeline))

test_dataloader = val_dataloader
