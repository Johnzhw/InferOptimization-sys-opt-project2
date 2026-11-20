#!/usr/bin/env bash
# 逐步复现：按研究链顺序执行（每步可单独运行）
#   bash run_stepwise.sh <step>
# 步骤：
#   0 单元测试        1 问题发现测量      2 接受率闸门（模型规模选型复核）
#   3 剖析组合        4 正确性对照        5 正式实验（同 run_all.sh）
#   6 汇总分析与图表  7 报告构建
set -eu
STEP="${1:-}"
ROOT=/home/johnzhw/vllm-study
LIB="$ROOT/reproduction_ewsjf_conjugate/scripts"
SUB="$ROOT/submission"

case "$STEP" in
0)
  cd /tmp && "$ROOT/.venv/bin/python" -m pytest "$SUB/code/tests" -q ;;
1)
  bash "$LIB/motivation_hol_8b.sh" ;;
2)
  bash /tmp/gate_8b.sh 2>/dev/null || echo "闸门脚本位于工作目录临时区，等效命令见 data/负载清单.md；亦可直接运行：vllm serve …（参数见 env/环境与配置清单.md）" ;;
3)
  bash "$LIB/profile_8b_combo.sh" ;;
4)
  bash "$LIB/correctness_8b.sh" ;;
5)
  bash "$SUB/scripts/run_all.sh" 1 ;;
6)
  cd /tmp && "$ROOT/.venv/bin/python" "$LIB/analyze_formal.py" \
    && "$ROOT/.venv/bin/python" "$SUB/report/make_submission_figures.py" ;;
7)
  cd /tmp && "$ROOT/.venv/bin/python" "$SUB/report/render_full_markdown.py" ;;
*)
  echo "用法: bash run_stepwise.sh <0-7>"; exit 1 ;;
esac
