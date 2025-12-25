mkdir -p ./save
mkdir -p ./trainlogs

# --- 自动选择空闲 GPU 的逻辑 ---
# 获取显存占用最低的 GPU ID
# nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 会返回每块卡的显存占用
# sort -n 按数值排序，head -n 1 取最小值，cut 获取对应的原始索引
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | cut -d ',' -f 1)

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
    --data-path /data2/huangxutao/projects/FairGrad/experiments/nyuv2/dataset \
    > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &