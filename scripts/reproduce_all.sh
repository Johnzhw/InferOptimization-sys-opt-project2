#!/usr/bin/env bash
# 一键复现：在任意满足前置条件的机器上运行全部负载 × 全部配置的实验，
# 自动创建结果目录并保存结果（bench JSON、服务日志、遥测、运行摘要、汇总表）。
#
# 必设环境变量（参见 复现指南.md）：
#   REPO_DIR      vllm 仓库目录（含本项目补丁，可编辑安装已完成）
#   TARGET_MODEL  目标模型目录（Qwen3-8B）
#   DRAFT_MODEL   草稿模型目录（Qwen3-0.6B）
# 可选：
#   REPRO_ROOT    工作区（默认 $PWD/repro_run）
#   GPU_ID        使用的卡号（默认 0）
#   PORT          服务端口（默认 8891）
#   SHAREGPT      ShareGPT JSON 路径（缺失时自动下载并校验 SHA256）
#   HF_ENDPOINT   模型/数据镜像（默认 https://hf-mirror.com）
#   SMOKE=1       冒烟模式（极小负载，验证链路）
#   CONFIGS / WORKLOADS  覆盖默认全集
set -eu

REPO_DIR="${REPO_DIR:?请设置 REPO_DIR 为 vllm 仓库目录}"
TARGET_MODEL="${TARGET_MODEL:?请设置 TARGET_MODEL}"
DRAFT_MODEL="${DRAFT_MODEL:?请设置 DRAFT_MODEL}"
REPRO_ROOT="${REPRO_ROOT:-$PWD/repro_run}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8891}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SMOKE="${SMOKE:-0}"
CONFIGS="${CONFIGS:-fcfs ewsjf spec ewsjf_spec}"
WORKLOADS="${WORKLOADS:-W1 W2 W3 W4}"

WL_DIR="$REPRO_ROOT/workloads"
RES_DIR="$REPRO_ROOT/results"
LOG_DIR="$REPRO_ROOT/logs"
mkdir -p "$WL_DIR" "$RES_DIR" "$LOG_DIR"
REG="$REPRO_ROOT/process_registry.jsonl"; touch "$REG"

echo "== [0/5] 环境检查 =="
command -v nvcc >/dev/null || { echo "错误：PATH 中无 nvcc（需 CUDA 工具链，见复现指南第 2 步）"; exit 1; }
NVCC="$(command -v nvcc)"
echo "nvcc: $NVCC"
nvcc --version | tail -1
nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv,noheader -i "$GPU_ID"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
[ "$USED" -le 500 ] || { echo "错误：GPU $GPU_ID 不空闲（${USED} MiB）"; exit 1; }
cd /tmp && "$REPO_DIR/../.venv/bin/python" -c "import vllm" 2>/dev/null \
  || python3 -c "import vllm" 2>/dev/null \
  || { echo "错误：vllm 不可导入（请先完成可编辑安装，见复现指南第 1 步）"; exit 1; }
PYOK=""
for P in "$REPO_DIR/../.venv/bin/python" python3; do
  if "$P" -c "import vllm" >/dev/null 2>&1; then PYOK="$P"; break; fi
done
echo "python: $PYOK"
VLLM_BIN="$(dirname "$PYOK")/vllm"
[ -x "$VLLM_BIN" ] || VLLM_BIN="$(dirname "$PYOK")/vllm"
echo "vllm: $VLLM_BIN"

echo "== [1/5] 负载准备 =="
SHAREGPT="${SHAREGPT:-$WL_DIR/ShareGPT_V3_unfiltered_cleaned_split.json}"
if [ ! -f "$SHAREGPT" ] && [ "$SMOKE" != "1" ]; then
  echo "下载 ShareGPT（约 640 MB）..."
  curl -L --retry 3 -o "$SHAREGPT" \
    "$HF_ENDPOINT/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
fi
if [ -f "$SHAREGPT" ]; then
  echo "ShareGPT SHA256:"; sha256sum "$SHAREGPT"
  echo "（基准值 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4）"
else
  echo "警告：无 ShareGPT，W1/W2 将跳过"
  WORKLOADS=$(echo "$WORKLOADS" | tr ' ' '\n' | grep -vE '^(W1|W2)$' | tr '\n' ' ')
fi
TARGET_MODEL="$TARGET_MODEL" OUT_DIR="$WL_DIR" SHAREGPT="$SHAREGPT" SMOKE="$SMOKE" \
  "$PYOK" "$(dirname "$0")/gen_workloads.py"

# 负载参数（正式冻结值；SMOKE 模式缩小规模）
wl_args() {
  case "$1" in
    W1) if [ "$SMOKE" = 1 ]; then echo "sharegpt|$SHAREGPT|--num-prompts 12 --request-rate 1.0";
        else echo "sharegpt|$SHAREGPT|--num-prompts 300 --request-rate 2.0"; fi ;;
    W2) if [ "$SMOKE" = 1 ]; then echo "custom|$WL_DIR/W2_sharegpt_long_output.jsonl|--custom-output-len -1 --num-prompts 8 --request-rate 0.5";
        else echo "custom|$WL_DIR/W2_sharegpt_long_output.jsonl|--custom-output-len -1 --num-prompts 400 --request-rate 0.5"; fi ;;
    W3) if [ "$SMOKE" = 1 ]; then echo "custom|$WL_DIR/W3_synthetic_bimodal.jsonl|--custom-output-len -1 --num-prompts 24 --request-rate 1.0";
        else echo "custom|$WL_DIR/W3_synthetic_bimodal.jsonl|--custom-output-len -1 --num-prompts 600 --request-rate 2.0"; fi ;;
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

stop_server() { # $1 pid $2 log
  if [ -d "/proc/$1" ]; then
    kill -TERM "$1" 2>/dev/null || true
    for i in $(seq 1 90); do [ -d "/proc/$1" ] || break; sleep 1; done
    [ -d "/proc/$1" ] && kill -KILL "$1" 2>/dev/null || true
  fi
  for i in $(seq 1 30); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
    [ "$U" -le 500 ] && break; sleep 5
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
    grep -cE "长度分区|scheduler active" "$OUT/server.log" >/dev/null 2>&1 || true
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
"$PYOK" "$(dirname "$0")/summarize_results.py" --results "$RES_DIR" \
  --out "$RES_DIR" || echo "（汇总脚本执行失败，原始 result.json 不受影响）"

echo "== [4/5] 完成 =="
echo "结果目录：$RES_DIR"
find "$RES_DIR" -name result.json | sort
