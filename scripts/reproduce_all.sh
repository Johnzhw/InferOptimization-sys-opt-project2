#!/usr/bin/env bash
# 一键复现：在任意满足前置条件的机器上运行全部负载 × 全部配置的实验
# 自动创建结果目录并保存结果（bench JSON、服务日志、遥测、运行摘要、汇总表）
#
# 必设环境变量：
#   TARGET_MODEL  目标模型目录（Qwen3-8B）
#   DRAFT_MODEL   草稿模型目录（Qwen3-0.6B）
# 可选：
#   REPRO_ROOT    工作区（默认 $PWD/repro_run）
#   GPU_ID        使用的卡号（默认 0）
#   PORT          服务端口（默认 8891）
#   HF_ENDPOINT   模型/数据镜像（默认 https://hf-mirror.com）
#   SMOKE=1       冒烟模式（极小负载，验证链路）
#   CONFIGS / WORKLOADS  覆盖默认全集
set -eu

# ---- 关键：用 BASH_SOURCE 解析绝对路径（不受后续 cd 影响） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"   # 仓库根目录

TARGET_MODEL="${TARGET_MODEL:?请设置 TARGET_MODEL（Qwen3-8B 目录）}"
DRAFT_MODEL="${DRAFT_MODEL:?请设置 DRAFT_MODEL（Qwen3-0.6B 目录）}"
REPRO_ROOT="${REPRO_ROOT:-$REPO_DIR/repro_run}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8891}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SMOKE="${SMOKE:-0}"
CONFIGS="${CONFIGS:-fcfs ewsjf spec ewsjf_spec}"
WORKLOADS="${WORKLOADS:-W1 W1_3p0 W2 W3 W3_1p6 W4}"

# 校验模型路径
[ -f "$TARGET_MODEL/config.json" ] || { echo "错误：TARGET_MODEL 无 config.json：$TARGET_MODEL"; exit 1; }
[ -f "$DRAFT_MODEL/config.json" ] || { echo "错误：DRAFT_MODEL 无 config.json：$DRAFT_MODEL"; exit 1; }
echo "目标模型: $TARGET_MODEL"
echo "草稿模型: $DRAFT_MODEL"

# 负载目录：优先用仓库自带的，缺失的从网上补
WL_DIR="$REPO_DIR/workloads"
# 结果目录：SMOKE 和全量分开存放，避免冒烟结果挡住正式实验
if [ "$SMOKE" = "1" ]; then
  RES_DIR="$REPRO_ROOT/results_smoke"
else
  RES_DIR="$REPRO_ROOT/results"
fi
mkdir -p "$RES_DIR"
REG="$REPRO_ROOT/process_registry.jsonl"; touch "$REG"

echo "== [0/5] 环境检查 =="
command -v nvcc >/dev/null || { echo "错误：PATH 中无 nvcc（需 CUDA 工具链）"; exit 1; }
echo "nvcc: $(command -v nvcc)"
nvcc --version | tail -1
nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv,noheader -i "$GPU_ID"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
[ "$USED" -le 500 ] || { echo "错误：GPU $GPU_ID 不空闲（${USED} MiB）"; exit 1; }

# 找到可用的 Python
PYOK=""
for P in "$REPO_DIR/.venv/bin/python" "$(which python3 2>/dev/null)"; do
  if [ -n "$P" ] && "$P" -c "import vllm" >/dev/null 2>&1; then PYOK="$P"; break; fi
done
[ -n "$PYOK" ] || { echo "错误：vllm 不可导入（请先 source .venv/bin/activate 或安装）"; exit 1; }

# 验证 torch 变体
"$PYOK" -c "
import torch
v = torch.__version__
if 'cu13' in v:
    print(f'错误：torch {v} 需要 CUDA 13 驱动')
    print('修复：uv pip install --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129')
    exit(1)
print(f'torch {v} CUDA={torch.version.cuda} 可用={torch.cuda.is_available()}')
" || exit 1

VLLM_BIN="$(dirname "$PYOK")/vllm"
echo "python: $PYOK"
echo "vllm: $VLLM_BIN"

echo "== [1/5] 负载准备 =="
# 合成负载（已预置于仓库 workloads/ 目录，直接复制到工作区）
for f in W2_sharegpt_long_output.jsonl W3_synthetic_bimodal.jsonl W4_synthetic_decode_heavy.jsonl; do
  if [ -f "$SCRIPT_DIR/../workloads/$f" ]; then
    cp -n "$SCRIPT_DIR/../workloads/$f" "$WL_DIR/$f" 2>/dev/null || true
  fi
done

# ShareGPT：仓库自带 500 条精简版（3.3MB），正式实验可指定全量文件
SHAREGPT_SUBSET="$REPO_DIR/workloads/ShareGPT_subset_500.json"
SHAREGPT_FULL="$WL_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ -n "${SHAREGPT:-}" ] && [ -f "$SHAREGPT" ]; then
  true  # 用户指定了路径，直接用
elif [ -f "$SHAREGPT_FULL" ]; then
  SHAREGPT="$SHAREGPT_FULL"  # 已下载全量
elif [ -f "$SHAREGPT_SUBSET" ]; then
  SHAREGPT="$SHAREGPT_SUBSET"
  echo "使用仓库自带 ShareGPT 精简版（500 条，3.3 MB）"
  echo "如需全量 94145 条，设置 SHAREGPT=<全量文件路径>"
else
  echo "下载 ShareGPT（约 643 MB）..."
  curl -L --retry 3 -o "$SHAREGPT_FULL" \
    "$HF_ENDPOINT/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
  SHAREGPT="$SHAREGPT_FULL"
fi
echo "ShareGPT: $SHAREGPT"

# 如合成负载仍缺失，在线生成
if [ ! -f "$WL_DIR/W3_synthetic_bimodal.jsonl" ] || [ ! -f "$WL_DIR/W4_synthetic_decode_heavy.jsonl" ]; then
  echo "生成合成负载..."
  TARGET_MODEL="$TARGET_MODEL" OUT_DIR="$WL_DIR" SHAREGPT="$SHAREGPT" SMOKE="$SMOKE" \
    "$PYOK" "$SCRIPT_DIR/gen_workloads.py"
fi

# 负载参数（含补充运行点）
wl_args() {
  case "$1" in
    W1) if [ "$SMOKE" = 1 ]; then echo "sharegpt|$SHAREGPT|--num-prompts 12 --request-rate 1.0";
        else echo "sharegpt|$SHAREGPT|--num-prompts 300 --request-rate 2.0"; fi ;;
    W1_3p0) if [ "$SMOKE" = 1 ]; then echo "sharegpt|$SHAREGPT|--num-prompts 12 --request-rate 2.0";
        else echo "sharegpt|$SHAREGPT|--num-prompts 300 --request-rate 3.0"; fi ;;
    W2) if [ "$SMOKE" = 1 ]; then echo "custom|$WL_DIR/W2_sharegpt_long_output.jsonl|--custom-output-len -1 --num-prompts 8 --request-rate 0.5";
        else echo "custom|$WL_DIR/W2_sharegpt_long_output.jsonl|--custom-output-len -1 --num-prompts 400 --request-rate 0.5"; fi ;;
    W3) if [ "$SMOKE" = 1 ]; then echo "custom|$WL_DIR/W3_synthetic_bimodal.jsonl|--custom-output-len -1 --num-prompts 24 --request-rate 1.0";
        else echo "custom|$WL_DIR/W3_synthetic_bimodal.jsonl|--custom-output-len -1 --num-prompts 600 --request-rate 2.0"; fi ;;
    W3_1p6) if [ "$SMOKE" = 1 ]; then echo "custom|$WL_DIR/W3_synthetic_bimodal.jsonl|--custom-output-len -1 --num-prompts 24 --request-rate 1.0";
        else echo "custom|$WL_DIR/W3_synthetic_bimodal.jsonl|--custom-output-len -1 --num-prompts 600 --request-rate 1.6"; fi ;;
    W4) if [ "$SMOKE" = 1 ]; then echo "custom|$WL_DIR/W4_synthetic_decode_heavy.jsonl|--custom-output-len -1 --num-prompts 12 --request-rate 0.5";
        else echo "custom|$WL_DIR/W4_synthetic_decode_heavy.jsonl|--custom-output-len -1 --num-prompts 400 --request-rate 0.8"; fi ;;
  esac
}

cfg_args() {
  case "$1" in
    fcfs)       echo "--scheduling-policy fcfs||" ;;
    ewsjf)      echo "--scheduling-policy ewsjf --ewsjf-max-queues 32 --ewsjf-strategic-interval 15|" ;;
    spec)       echo "|{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":4}" ;;
    ewsjf_spec) echo "--scheduling-policy ewsjf --ewsjf-max-queues 32 --ewsjf-strategic-interval 15|{\"method\":\"draft_model\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":4}" ;;
  esac
}

start_server() { # $1 cfg $2 log
  local CFGARGS SPECARGS; IFS='|' read -r CFGARGS SPECARGS <<< "$(cfg_args "$1")"
  bash -c '
    export CUDA_VISIBLE_DEVICES='"$GPU_ID"'
    export VLLM_CACHE_ROOT='"$REPRO_ROOT"'/.vllm_cache
    export VLLM_SERVER_DEV_MODE=1
    export HF_HOME='"$REPRO_ROOT"'/.hf_home
    export HF_ENDPOINT='"$HF_ENDPOINT"'
    export CUDA_HOME='"${CUDA_HOME:-/usr/local/cuda-12.8}"'
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    exec '"$VLLM_BIN"' serve "'"$TARGET_MODEL"'" \
      --host 127.0.0.1 --port '"$PORT"' '"$CFGARGS"' \
      '"$( [ -n "$SPECARGS" ] && echo "--speculative-config '$SPECARGS'" || true )"' \
      --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.92
  ' > "$2" 2>&1 &
  echo $!
}

stop_server() { # $1 pid
  if [ -d "/proc/$1" ]; then
    kill -TERM "$1" 2>/dev/null || true
    for i in $(seq 1 90); do [ -d "/proc/$1" ] || break; sleep 1; done
    [ -d "/proc/$1" ] && kill -KILL "$1" 2>/dev/null || true
  fi
  # 等端口释放
  for i in $(seq 1 60); do
    if ! exec 3<>"/dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then break; fi
    exec 3>&- 2>/dev/null || true; sleep 1
  done
  printf '{"ts":"%s","event":"stop","pid":%s}\n' "$(date -Is)" "$1" >> "$REG"
}

echo "== [2/5] 运行实验矩阵 =="
for CFG in $CONFIGS; do
  for WL in $WORKLOADS; do
    OUT="$RES_DIR/$CFG/$WL"
    [ -f "$OUT/result.json" ] && { echo "跳过已有 $CFG/$WL"; continue; }
    mkdir -p "$OUT"
    echo "---- $CFG / $WL ----"
    IFS='|' read -r DN DP EXTRA <<< "$(wl_args "$WL")"
    PID=$(start_server "$CFG" "$OUT/server.log")
    printf '{"ts":"%s","event":"start","pid":%s,"port":%s,"cfg":"%s","wl":"%s"}\n' \
      "$(date -Is)" "$PID" "$PORT" "$CFG" "$WL" >> "$REG"
    OK=0
    for i in $(seq 1 240); do
      curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200 && { OK=1; break; }
      sleep 3
    done
    if [ "$OK" != 1 ]; then
      echo "服务启动失败，日志尾部："; tail -20 "$OUT/server.log"; stop_server "$PID"
      exit 1
    fi
    grep -E "GPU KV cache size" "$OUT/server.log" | head -1 | tee "$OUT/capacity.txt"
    curl -s -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null || true
    cd /tmp && "$VLLM_BIN" bench serve \
      --base-url "http://127.0.0.1:$PORT" \
      --dataset-name "$DN" --dataset-path "$DP" $EXTRA \
      --burstiness 1.0 --seed 20261101 \
      --save-result --save-detailed \
      --result-dir "$OUT" --result-filename result.json \
      --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 2>&1 | tail -5 | tee "$OUT/bench_tail.txt"
    grep -E "strategic tick|tactical telemetry|SpecDecoding metrics" "$OUT/server.log" \
      | tail -4 > "$OUT/mechanism_telemetry.txt" || true
    stop_server "$PID"
    printf '{"config":"%s","workload":"%s","finished":"%s"}\n' "$CFG" "$WL" "$(date -Is)" \
      > "$OUT/run_summary.json"
    echo "完成 $CFG/$WL -> $OUT"
  done
done

echo "== [3/5] 汇总 =="
"$PYOK" "$SCRIPT_DIR/summarize_results.py" --results "$RES_DIR" \
  --out "$RES_DIR" || echo "（汇总脚本执行失败）"
echo ""
"$PYOK" "$SCRIPT_DIR/print_tables.py" "$RES_DIR" || true

echo "== [4/5] 完成 =="
echo "结果目录：$RES_DIR"
find "$RES_DIR" -name result.json | sort
