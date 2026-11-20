#!/usr/bin/env python3
"""Summarize reproduce_all.sh results into summary.csv / comparison.csv."""
import argparse
import csv
import glob
import json
import os

CFG_LABEL = {
    "fcfs": "先来先服务基线配置",
    "ewsjf": "长度分区加权调度器配置",
    "spec": "草稿模型投机解码配置",
    "ewsjf_spec": "长度分区加权调度器与草稿模型投机解码组合配置",
}


def stats(v):
    if not v:
        return None, None, None
    v = sorted(v)
    n = len(v)
    return sum(v) / n, v[int(n * 0.95)], v[min(n - 1, int(n * 0.99))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(f"{args.results}/*/*/result.json")):
        parts = p.split(os.sep)
        cfg, wl = parts[-3], parts[-2]
        d = json.load(open(p))
        rec = {
            "config": CFG_LABEL.get(cfg, cfg), "config_key": cfg, "workload": wl,
            "completed": d.get("completed"), "failed": d.get("failed"),
            "request_throughput": d.get("request_throughput"),
            "output_throughput": d.get("output_throughput"),
            "mean_ttft_ms": d.get("mean_ttft_ms"),
            "p99_ttft_ms": d.get("p99_ttft_ms"),
            "mean_tpot_ms": d.get("mean_tpot_ms"),
            "p99_itl_ms": d.get("p99_itl_ms"),
            "spec_acceptance_length": d.get("spec_decode_acceptance_length"),
        }
        if "input_lens" in d and "ttfts" in d:
            il = d["input_lens"]
            tt = [t * 1000 for t in d["ttfts"]]
            short = [t for l, t in zip(il, tt) if l <= 512]
            m, p95, p99 = stats(short)
            rec.update({"short_ttft_mean_ms": m, "short_ttft_p95_ms": p95,
                        "short_ttft_p99_ms": p99})
        rows.append(rec)
    if not rows:
        print("无结果可汇总")
        return

    cols = ["workload", "config_key", "config", "completed", "failed",
            "request_throughput", "output_throughput", "mean_ttft_ms",
            "p99_ttft_ms", "short_ttft_mean_ms", "short_ttft_p95_ms",
            "short_ttft_p99_ms", "mean_tpot_ms", "p99_itl_ms",
            "spec_acceptance_length"]
    with open(f"{args.out}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 3)
                            if isinstance(r.get(k), float) else r.get(k))
                        for k in cols})

    def avg(cfg, wl, key):
        vals = [r[key] for r in rows
                if r["config_key"] == cfg and r["workload"] == wl
                and r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def pct(o, b):
        if o is None or b in (None, 0):
            return "N/A"
        return f"{(o - b) / b * 100:+.1f}%"

    comp = []
    wls = sorted({r["workload"] for r in rows})
    pairs = [("ewsjf", "fcfs"), ("spec", "fcfs"), ("ewsjf_spec", "fcfs"),
             ("ewsjf_spec", "spec"), ("ewsjf_spec", "ewsjf")]
    for wl in wls:
        for oc, bc in pairs:
            for key in ["request_throughput", "output_throughput",
                        "mean_ttft_ms", "short_ttft_p99_ms", "mean_tpot_ms"]:
                b, o = avg(bc, wl, key), avg(oc, wl, key)
                comp.append({
                    "workload": wl,
                    "comparison": f"{CFG_LABEL[oc]} vs {CFG_LABEL[bc]}",
                    "metric": key,
                    "baseline": round(b, 3) if b is not None else "N/A",
                    "optimized": round(o, 3) if o is not None else "N/A",
                    "delta": round(o - b, 3) if None not in (o, b) else "N/A",
                    "change": pct(o, b) if None not in (o, b) else "N/A",
                })
    with open(f"{args.out}/comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["workload", "comparison", "metric",
                                          "baseline", "optimized", "delta",
                                          "change"])
        w.writeheader()
        w.writerows(comp)
    print(f"summary.csv（{len(rows)} 运行）与 comparison.csv（{len(comp)} 行）"
          f"已写入 {args.out}")


if __name__ == "__main__":
    main()
