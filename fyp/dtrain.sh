
# Active learning
# UI
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/al_ui.yaml

# Random
# CUDA_VISIBLE_DEVICES=1 python train_dual_active.py --config yaml_config/al_random.yaml

# MaxEntropy
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/al_maxentropy.yaml

# MaxEntropy + Distance
CUDA_VISIBLE_DEVICES=1 python train_dual_active.py --config yaml_config/al_maxentropy_dist.yaml



# active learning - demo
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/al_demo.yaml



## Single training - demo
# CUDA_VISIBLE_DEVICES=0 python train_dual.py --config yaml_config/train_resnext.yaml
