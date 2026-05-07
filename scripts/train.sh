# GaussTR-FeatUp, single GPU
PYTHONPATH=. mim train mmdet3d configs/gausstr_featup.py -G 1

# GaussTR-FeatUp, multi GPU
PYTHONPATH=. mim train mmdet3d configs/gausstr_featup.py -l pytorch -G 8

# GaussTR-Talk2DINO, single GPU
PYTHONPATH=. mim train mmdet3d configs/gausstr_talk2dino.py -G 1

# GaussTR-Talk2DINO, multi GPU
PYTHONPATH=. mim train mmdet3d configs/gausstr_talk2dino.py -l pytorch -G 8

# 附加参数
    --work-dir work_dirs/gausstr_talk2dino_chunks_exp1