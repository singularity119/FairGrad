# --- 1. 定义 AutoDL 输出路径 ---
EXP_ROOT="/root/autodl-tmp/experiment/quantum_chemistry_experiment"
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

method=cagrad # famo模式切换
alpha=2.0
seed=0
gamma=0.001 # 增加gamma用于famo模式切换

# --- 4. 运行训练 ---
# 注意：我们增加了 --save-dir 参数来指定输出位置
nohup python -u trainer.py \
    --method=$method \
    --alpha=$alpha \
    --seed=$seed \
    --scale-y=True \
    --data-path "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    > "$LOG_DIR/$method-alpha$alpha-sd$seed.log" 2>&1 &

echo "Training started. Logs: $LOG_DIR/$method-alpha$alpha-sd$seed.log"
