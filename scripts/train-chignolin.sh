cd ../

CUDA_VISIBLE_DEVICES=$1 python train.py \
  --config configs/chignolin/default.yaml