# --- 1. 定义 AutoDL 输出路径 ---
EXP_ROOT="/root/autodl-tmp/experiment/cityscapes_experiment"
SAVE_DIR="$EXP_ROOT/save"
LOG_DIR="$EXP_ROOT/trainlogs"

# --- 2. 创建目录 ---
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# --- 3. 环境与路径设置 ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/root/FairGrad
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

method=fairgrad
alpha=2.0
seed=1        # 师兄建议先试种子 1
sigma=1e-4    # 扰动强度

# --- 4. 运行训练 (循环运行 5 次不同扰动的实验) ---
# 我们使用不同的 noise-seed (101 到 105) 来产生不同的随机扰动
for ns in 101 102 103 104 105
do
    echo "Starting experiment with seed=$seed, noise-seed=$ns..."
    
    # 注意：这里去掉了末尾的 &，改为顺序执行，防止显存溢出。
    # 如果你想同时后台跑，请确保显存足够并加上 &。
    python -u trainer_noise.py \
        --method=$method \
        --seed=$seed \
        --alpha=$alpha \
        --perturb-sigma=$sigma \
        --noise-seed=$ns \
        --gpu 0 \
        --data-path /root/autodl-tmp/dataset/cityscapes \
        --save-dir "$SAVE_DIR" \
        > "$LOG_DIR/$method-alpha$alpha-sd$seed-ns$ns.log" 2>&1
        
    echo "Finished experiment with noise-seed=$ns. Results in $LOG_DIR"
done

echo "All 5 perturbation experiments for seed $seed are completed."
