#!/bin/bash

# 脚本名称: run_node_evaluation.sh
# 功能: 对不同节点数的平面图进行评测并保存结果

# 设置时间戳格式
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="planar_node_evaluation_${TIMESTAMP}.log"

# 确保在项目根目录执行
cd "$(dirname "$0")" || exit
pwd
# 定义开始和结束节点数
START_NODE=64
END_NODE=136
STEP=8

# 创建日志文件并记录开始时间
echo "==============================================" > "$LOG_FILE"
echo "开始执行平面图评测 - 时间: $(date)" >> "$LOG_FILE"
echo "节点范围: $START_NODE 到 $END_NODE，步长: $STEP" >> "$LOG_FILE"
echo "使用checkpoint: /home/ly/max/checkpoint/planar-59999-14h.ckpt" >> "$LOG_FILE"
echo "使用已生成的数据集，无需重新生成" >> "$LOG_FILE"
echo "==============================================" >> "$LOG_FILE"

# 记录函数 - 同时输出到控制台和日志文件
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 确保data/my_planar目录不存在（如果存在先备份）
if [ -d "data/my_planar" ]; then
    log "检测到data/my_planar目录存在，将其备份..."
    mv data/my_planar "data/my_planar_backup_$(date +%s)"
fi

# 循环评测不同节点数
for ((node_num=START_NODE; node_num<END_NODE; node_num+=STEP)); do
    log "开始处理节点数为 $node_num 的数据集..."
    
    # 检查对应的数据集目录是否存在
    if [ ! -d "data/my_planar_${node_num}" ]; then
        log "错误: data/my_planar_${node_num} 目录不存在，请先生成该节点数的数据集!"
        continue
    fi
    
    # 重命名数据集目录以供使用
    log "重命名 data/my_planar_${node_num} 为 data/my_planar..."
    mv "data/my_planar_${node_num}" data/my_planar
    
    # 执行评测
    log "在src目录中执行评测..."
    # 使用当前时间戳作为随机种子，确保每次运行都有不同的随机性
    RANDOM_SEED=$((RANDOM % 1000000))
    (cd ~/max/DeFoG-swanlab && PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=3 python src/main.py +experiment=planar dataset=my_planar general.test_only=/home/ly/max/checkpoints/planar-59999-14h.ckpt train.seed=$RANDOM_SEED >> "../../$LOG_FILE" 2>&1)
    if [ $? -ne 0 ]; then
        log "错误: 评测执行失败!"
        # 即使评测失败也要恢复目录名
        mv data/my_planar "data/my_planar_${node_num}"
        continue
    fi
    
    # 评测完成后恢复原目录名
    log "评测完成，恢复目录名为 data/my_planar_${node_num}..."
    mv data/my_planar "data/my_planar_${node_num}"
    
    log "完成节点数为 $node_num 的评测"
    echo "==============================================" >> "$LOG_FILE"
done

log "全部评测完成!"
echo "评测日志已保存至: $LOG_FILE" 