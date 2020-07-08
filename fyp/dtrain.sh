# single training - demo
# CUDA_VISIBLE_DEVICES=0 python train_dual.py --config yaml_config/new_dual_resnext.yaml

# active learning
# UI
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/new_al_random_resnext.yaml

# Random
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/new_al_ui_resnext.yaml

# active learning - demo
# CUDA_VISIBLE_DEVICES=0 python train_dual_active.py --config yaml_config/demo_new_al_resnext.yaml