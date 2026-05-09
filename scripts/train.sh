# GaussTR-FeatUp, multi GPU
PYTHONPATH=. mim train mmdet3d configs/gausstr_featup_chunks.py -l pytorch -G 8 \
--work-dir work_dirs/gausstr_featup_chunks \

# GaussTR-Talk2DINO, multi GPU
PYTHONPATH=. mim train mmdet3d configs/gausstr_talk2dino_chunks.py -l pytorch -G 8 \
--work-dir work_dirs/gausstr_talk2dino_chunks \

# 附加参数
--cfg-options model.num_queries=600 \