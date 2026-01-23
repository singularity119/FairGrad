# --- 1. 定义 AutoDL 输出路径 ---
EXP_ROOT="/root/autodl-tmp/experiment/quantum_chemistry_experiment/apt"
DATA_DIR="/root/autodl-tmp/dataset/qm9"
SAVE_DIR="$EXP_ROOT/save"
LOG_DIR="$EXP_ROOT/trainlogs"

# --- 2. 创建目录 ---
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"

# --- 3. GPU 设置 ---
export CUDA_VISIBLE_DEVICES=0
# 解释：将项目根目录 /root/FairGrad 添加到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/root/FairGrad
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

method=fairgrad
alpha=2.0
seed=1
beta_range="0.1-0.9"

# --- 4. 运行训练 ---
nohup python -u trainer_apt.py \
    --method=$method \
    --alpha=$alpha \
    --seed=$seed \
    --scale-y=True \
    --beta-range ${beta_range//-/ } \
    --data-path "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    > "$LOG_DIR/$method-alpha$alpha-sd$seed-beta_range$beta_range.log" 2>&1 &

echo "Training started. Logs: $LOG_DIR/$method-alpha$alpha-sd$seed-beta_range$beta_range.log"
