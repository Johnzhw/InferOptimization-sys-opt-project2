#!/usr/bin/env bash
# 一键复现：正式实验全矩阵（断点续跑，已有 result.json 的运行自动跳过）。
# 前置条件：
#   1. 工作目录 /home/johnzhw/vllm-study（vLLM v0.28.0 editable 安装于 .venv，源码 ./vllm）
#   2. 本文件夹 code/patches/ 的九个补丁已按 INTEGRATION.md 应用到 ./vllm
#   3. GPU 空闲（脚本每次启动前做只读检查，默认用 7 号卡，可用 GPU=n 覆盖）
# 用法： bash run_all.sh [重复轮数，默认 1]
set -u
export GPU="${GPU:-7}"
WLS=(W3 W4 W1 W2)
CONFIGS=(fcfs ewsjf spec ewsjf_spec)
REPS="${1:-1}"
ROOT=/home/johnzhw/vllm-study
LIB="$ROOT/reproduction_ewsjf_conjugate/scripts"

for rep in $(seq 1 "$REPS"); do
  for wl in "${WLS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      bash "$LIB/run_one.sh" "$cfg" "$wl" "$rep" || { echo "FAILED at $cfg/$wl/$rep"; exit 1; }
    done
  done
done
# 补充运行点（饱和邻域两轮 + 动机测量点）
bash "$LIB/run_supplementary.sh"
bash "$LIB/run_rep2_w1_3p0.sh"
echo ALL-DONE
