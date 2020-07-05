# CUDA_VISIBLE_DEVICES=0 python evaluate_torch.py
# CUDA_VISIBLE_DEVICES=0 python dual_train.py --config yaml_config/dual_xception.yaml
# CUDA_VISIBLE_DEVICES=0 python td_new.py
CUDA_VISIBLE_DEVICES=1 python td_new.py
# CUDA_VISIBLE_DEVICES=1 python test_torch.py