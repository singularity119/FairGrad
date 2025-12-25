mkdir -p ./save
mkdir -p ./trainlogs

export CUDA_VISIBLE_DEVICES=7


echo "Selected GPU: $CUDA_VISIBLE_DEVICES"

method=cagrad
alpha=2.0
seed=0

# nohup python -u trainer.py --method=$method --seed=$seed --alpha=$alpha > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &

nohup python -u trainer.py \
    --method=$method \
    --seed=$seed \
    --alpha=$alpha \
    --gpu 0 \
    --data-path /data2/huangxutao/projects/FairGrad/experiments/cityscapes/dataset \
    > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &