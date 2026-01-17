# --- 1. 定义 AutoDL 输出路径 ---
EXP_ROOT="/root/autodl-tmp/experiment/cityscapes_experiment"
SAVE_DIR="$EXP_ROOT/save/init"
LOG_DIR="$EXP_ROOT/trainlogs/init"
FINETUNING_LOG_DIR="$LOG_DIR/finetuning"

# --- 2. 创建目录 ---
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$FINETUNING_LOG_DIR"

# --- 3. 环境与路径设置 ---
export CUDA_VISIBLE_DEVICES=0
# 确保项目根目录在 PYTHONPATH 中
export PYTHONPATH=$PYTHONPATH:/root/FairGrad
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

method=fairgrad
alpha=2.0
# 可选: rbd_multitask_curvature_fullspace 或留空表示 None
init_type=rbd_multitask_curvature_fullspace 
steps=50
seed=0
h_strategy="cone"
angle=0.34
timestamp=$(date +"%Y%m%d_%H%M%S")

if [ -z "$init_type" ] || [ "$init_type" == "None" ]; then
    # 情况 A: 不使用初始化
    echo "No initialization type specified or None. Starting standard training."
    nohup python -u trainer_init.py \
        --method=$method \
        --seed=$seed \
        --alpha=$alpha \
        --steps=$steps \
        --init_type=None \
        --gpu 0 \
        --data-path /root/autodl-tmp/dataset/cityscapes2 \
        --save-dir "$SAVE_DIR" \
        > "$FINETUNING_LOG_DIR/$method-alpha$alpha-no-init-sd$seed-$timestamp.log" 2>&1 &
else
    # 情况 B: 使用指定的初始化类型
    echo "Starting training with initialization: $init_type"
    nohup python -u trainer_init.py \
        --method=$method \
        --seed=$seed \
        --init_type=$init_type \
        --alpha=$alpha \
        --steps=$steps \
        --gpu 0 \
        --data-path /root/autodl-tmp/dataset/cityscapes2 \
        --save-dir "$SAVE_DIR" \
        > "$FINETUNING_LOG_DIR/$method-alpha$alpha-sd$seed-h_$h_strategy-angle$angle-$init_type.log" 2>&1 &
fi

echo "Training started. Logs: $FINETUNING_LOG_DIR"
