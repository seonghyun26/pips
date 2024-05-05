cd ../

echo "Training all models"
echo

echo ">> Alanine potential"
CUDA_VISIBLE_DEVICES=2 python train.py \
  --config configs/alanine/potential.yaml &
sleep 1

echo ">> Alanine force"
CUDA_VISIBLE_DEVICES=3 python train.py \
  --config configs/alanine/force.yaml &
sleep 1

echo ">> Poly potential"
CUDA_VISIBLE_DEVICES=4 python train.py \
  --config configs/poly/potential.yaml &
sleep 1

echo ">> Poly force"
CUDA_VISIBLE_DEVICES=5 python train.py \
  --config configs/poly/force.yaml &
sleep 1

echo ">> Chignolin potential"
CUDA_VISIBLE_DEVICES=6 python train.py \
  --config configs/chignolin/potential.yaml &
sleep 1

echo ">> Chignolin force"
CUDA_VISIBLE_DEVICES=7 python train.py \
  --config configs/chignolin/force.yaml &