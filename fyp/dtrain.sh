# single training - demo
# CUDA_VISIBLE_DEVICES=0 python train_dual.py --config yaml_config/new_dual_resnext.yaml

# active learning
CUDA_VISIBLE_DEVICES=0 python dual_active.py --config al_dual_xception.yaml
