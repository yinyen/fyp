
# Active learning
# Random
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/al_random.yaml

# UI
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/al_ui.yaml

# active learning - demo
CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/al_demo.yaml


## Single training - demo
# CUDA_VISIBLE_DEVICES=0 python train_dual.py --config yaml_config/train_resnext.yaml
