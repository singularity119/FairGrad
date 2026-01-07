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
noise_seed=101     #101 102 103 104 105


echo "Starting experiment with seed=$seed, noise-seed=$noise_seed..."
    
    # 注意：这里去掉了末尾的 &，改为顺序执行，防止显存溢出。
    # 如果你想同时后台跑，请确保显存足够并加上 &。
nohup python -u trainer_noise.py \
        --method=$method \
        --seed=$seed \
        --alpha=$alpha \
        --perturb-sigma=$sigma \
        --noise-seed=$noise_seed \
        --gpu 0 \
        --data-path /root/autodl-tmp/dataset/cityscapes2 \
        --save-dir "$SAVE_DIR" \
        > "$LOG_DIR/$method-alpha$alpha-sd$seed-ns$noise_seed.log" 2>&1 &
        
echo "Finished experiment with noise-seed=$noise_seed. Results in $LOG_DIR"

