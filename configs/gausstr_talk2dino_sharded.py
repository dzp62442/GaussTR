_base_ = './gausstr_talk2dino.py'

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
        final_dim=(504, 896),
        resize_lim=[0.56, 0.56],
        is_train=True),
    dict(
        type='LoadShardedFeatMaps',
        _scope_='mmdet3d',
        group='depth_metric3d',
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

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='NuScenesOccShardedDataset',
        shard_root='data/gausstr_shards',
        split='train',
        preload_mode='lazy',
        max_cache_bytes=32 * 1024**3,
        prefetch_shards=1,
        prefetch_workers=1,
        serialize_data=False,
        required_groups=dict(raw='raw_nuscenes', depth='depth_metric3d'),
        pipeline=train_pipeline))
