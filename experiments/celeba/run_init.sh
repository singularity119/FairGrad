# --- 1. 定义 AutoDL 输出路径 ---
EXP_ROOT="/root/autodl-tmp/experiment/celeba_experiment"
SAVE_DIR="$EXP_ROOT/save/init"
LOG_DIR="$EXP_ROOT/trainlogs/init"
FINETUNING_LOG_DIR="$LOG_DIR/finetuning"

# --- 2. 创建目录 ---
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$FINETUNING_LOG_DIR"

# --- 3. 环境与路径设置 ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/root/FairGrad
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

method=fairgrad
alpha=2.0
init_type=rbd_multitask_curvature_clustering_once # 设置为空字符串表示跳过初始化
steps=0
timestamp=$(date +"%Y%m%d_%H%M%S")

if [ -z "$init_type" ]; then
    # 没有初始化类型
    echo "No initialization type specified. Skipping initialization."
    # 这里为了示例只保留一个启动命令，你可以根据需要开启更多
    seed=42
    nohup python -u trainer_init.py \
        --method=$method --seed=$seed --alpha=$alpha --steps=$steps \
        --gpu 0 --data-path /root/autodl-tmp/dataset/celeba --save-dir "$SAVE_DIR" \
        > "$FINETUNING_LOG_DIR/$method-alpha$alpha-no-init-sd$seed-steps$steps-$timestamp.log" 2>&1 &
else
    # 有初始化类型
    seed=2
    nohup python -u trainer_init.py \
        --method=$method --seed=$seed --init_type=$init_type --alpha=$alpha --steps=$steps \
        --gpu 0 --data-path /root/autodl-tmp/dataset/celeba --save-dir "$SAVE_DIR" \
        > "$FINETUNING_LOG_DIR/$method-alpha$alpha-$init_type-sd$seed-steps$steps-$timestamp.log" 2>&1 &
fi

echo "Initialization training started. Logs in $LOG_DIR"
