#!/usr/bin/env python3
"""Generate the synthetic workloads (W2/W3/W4) with the target-model tokenizer.

Env:
  TARGET_MODEL  (必填) 目标模型目录（含 tokenizer）
  OUT_DIR       (默认 ./workloads) 输出目录
  SHAREGPT      (可选) ShareGPT JSON 路径，提供时生成 W2 长输出子集
  SMOKE         (=1 时生成极小规模，用于链路冒烟)
"""
import json
import os
import random
import sys

TARGET = os.environ.get("TARGET_MODEL")
OUT = os.environ.get("OUT_DIR", "./workloads")
SHAREGPT = os.environ.get("SHAREGPT")
SMOKE = os.environ.get("SMOKE", "0") == "1"

if not TARGET:
    sys.exit("请设置 TARGET_MODEL 为目标模型目录")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(
    os.environ.get("VENV_SITE", ""), ""))  # optional hook
from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained(TARGET)

WORDS = ("the quick brown fox jumps over lazy dog system scheduler request "
         "model server inference token cache memory batch queue latency prompt "
         "compute kernel stream draft verify accept reject step round time "
         "service workload arrival rate length short long mixture content").split()


def make_prompt(target_len: int, rng: random.Random) -> str:
    parts = []
    n = max(4, target_len // 2 + 8)
    while True:
        text = "Summarize the following notes: " + " ".join(
            rng.choice(WORDS) for _ in range(n))
        L = len(TOK(text).input_ids)
        if abs(L - target_len) <= 8:
            return text
        if L < target_len - 8:
            n += max(1, (target_len - L) // 2)
        else:
            n -= max(1, (L - target_len) // 2)


def w3():
    n = 24 if SMOKE else 600
    rng = random.Random(20260201)
    rows = []
    for i in range(n):
        long_one = (i % 5 == 4)
        if long_one:
            plen, olen = rng.randint(2048, 4096), rng.randint(96, 256)
        else:
            plen, olen = rng.randint(32, 256), rng.randint(32, 128)
        if SMOKE:
            plen, olen = min(plen, 96), 24
        rows.append({"prompt": make_prompt(plen, rng), "output_tokens": olen,
                     "meta": {"kind": "long" if long_one else "short"}})
    with open(f"{OUT}/W3_synthetic_bimodal.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"W3 -> {n} rows")


def w4():
    n = 12 if SMOKE else 400
    rng = random.Random(20260202)
    rows = []
    for _ in range(n):
        plen = rng.randint(64, 192)
        rows.append({"prompt": make_prompt(plen, rng),
                     "output_tokens": 48 if SMOKE else 320,
                     "meta": {"kind": "decode_heavy"}})
    with open(f"{OUT}/W4_synthetic_decode_heavy.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"W4 -> {n} rows")


def w2():
    if not SHAREGPT or not os.path.exists(SHAREGPT):
        print("SHAREGPT 未提供，跳过 W2 生成")
        return
    data = json.load(open(SHAREGPT))
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    random.Random(20261102).shuffle(data)
    target_n = 12 if SMOKE else 400
    rows = []
    for d in data:
        if len(rows) >= target_n:
            break
        prompt = d["conversations"][0]["value"]
        completion = d["conversations"][1]["value"]
        plen = len(TOK(prompt).input_ids)
        olen = len(TOK(completion).input_ids)
        if plen < 16 or plen > 4096 or olen < 256 or olen > 1024:
            continue
        rows.append({"prompt": prompt, "output_tokens": olen})
    with open(f"{OUT}/W2_sharegpt_long_output.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"W2 -> {len(rows)} rows")


if __name__ == "__main__":
    w3()
    w4()
    w2()
