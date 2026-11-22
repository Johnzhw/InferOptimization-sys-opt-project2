#!/usr/bin/env python3
"""按负载分组打印格式化结果表（与报告总表对应，每个负载一张表）。"""
import glob
import json
import os
import sys

CFG_LABEL = {
    "fcfs": "先来先服务基线",
    "ewsjf": "长度分区加权调度器",
    "spec": "草稿模型投机解码",
    "ewsjf_spec": "组合配置",
}
WL_LABEL = {
    "W1": "W1 ShareGPT 混合",
    "W1_3p0": "W1@3.0 饱和邻域",
    "W2": "W2 长输出子集",
    "W3": "W3 双峰混合",
    "W3_1p6": "W3@1.6 动机测量点",
    "W4": "W4 解码重负载",
}


def fmt(v, suffix="", width=8):
    if v is None:
        return "N/A".rjust(width)
    if isinstance(v, float):
        return f"{v:.1f}{suffix}".rjust(width)
    return str(v).rjust(width)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RES_DIR", "results")
    rows = []
    # 支持两种目录结构：
    #   <dir>/<cfg>/<wl>/result.json          （reproduce_all.sh 输出）
    #   <dir>/<cfg>/<wl>/<repN>/result.json  （多轮重复实验输出）
    paths = sorted(glob.glob(f"{results_dir}/*/*/result.json"))
    paths += sorted(glob.glob(f"{results_dir}/*/*/*/result.json"))
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        parts = p.split(os.sep)
        if parts[-2].startswith("rep"):
            cfg, wl = parts[-4], parts[-3]  # repN 结构
        else:
            cfg, wl = parts[-3], parts[-2]  # 直接结构
        d = json.load(open(p))
        il = d.get("input_lens") or []
        tt = [t * 1000 for t in (d.get("ttfts") or [])]
        short = sorted(t for l, t in zip(il, tt) if l <= 512)
        rows.append({
            "cfg": cfg, "wl": wl,
            "done": d.get("completed", 0),
            "rps": d.get("request_throughput"),
            "ott": d.get("output_throughput"),
            "ttft_mean": d.get("mean_ttft_ms"),
            "ttft_p95": d.get("p95_ttft_ms"),
            "ttft_p99": d.get("p99_ttft_ms"),
            "sp95": short[int(len(short) * .95)] if short else None,
            "sp99": short[min(len(short) - 1, int(len(short) * .99))] if short else None,
            "tpot": d.get("mean_tpot_ms"),
            "acc": d.get("spec_decode_acceptance_length"),
        })

    if not rows:
        print("无结果可展示"); return

    wls = sorted({r["wl"] for r in rows}, key=lambda w: list(WL_LABEL.keys()).index(w) if w in WL_LABEL else 99)

    # 每个负载一张表
    for wl in wls:
        wl_rows = [r for r in rows if r["wl"] == wl]
        if not wl_rows:
            continue
        print(f"\n{'='*100}")
        print(f"  {WL_LABEL.get(wl, wl)}")
        print(f"{'='*100}")
        header = f"  {'配置':<20} {'完成':>5} {'req/s':>7} {'tok/s':>8} {'TTFT均值':>9} {'TTFT p95':>9} {'TTFT p99':>9} {'短p95':>8} {'短p99':>8} {'TPOT':>7} {'接受长':>6}"
        print(header)
        print(f"  {'-'*97}")
        for r in sorted(wl_rows, key=lambda x: ["fcfs", "ewsjf", "spec", "ewsjf_spec"].index(x["cfg"])):
            acc = f"{r['acc']:.2f}" if r["acc"] else "  —"
            print(f"  {CFG_LABEL.get(r['cfg'], r['cfg']):<20} {r['done']:>5} "
                  f"{fmt(r['rps']):>7} {fmt(r['ott']):>8} "
                  f"{fmt(r['ttft_mean'], 'm'):>9} {fmt(r['ttft_p95'], 'm'):>9} {fmt(r['ttft_p99'], 'm'):>9} "
                  f"{fmt(r['sp95'], 'm'):>8} {fmt(r['sp99'], 'm'):>8} "
                  f"{fmt(r['tpot'], 'm'):>7} {acc:>6}")

        # 配对比较
        def avg(cfg, key):
            vals = [r[key] for r in wl_rows if r["cfg"] == cfg and r.get(key)]
            return sum(vals) / len(vals) if vals else None
        pairs = [("spec", "fcfs"), ("ewsjf_spec", "fcfs"), ("ewsjf_spec", "spec")]
        has_comp = False
        for oc, bc in pairs:
            if not any(r["cfg"] == oc for r in wl_rows) or not any(r["cfg"] == bc for r in wl_rows):
                continue
            if not has_comp:
                print(f"\n  关键对比（负差值=改善）：")
                print(f"  {'比较':<44} {'TPOT变化':>10} {'短p95变化':>10} {'吞吐变化':>10}")
                print(f"  {'-'*77}")
                has_comp = True
            for key, label in [("tpot", "TPOT"), ("sp95", "短p95"), ("ott", "吞吐")]:
                b, o = avg(bc, key), avg(oc, key)
                if b and o:
                    pct = (o - b) / b * 100
                    print(f"  {CFG_LABEL[oc]:>20} vs {CFG_LABEL[bc]:<20} {label:>8} "
                          f"{pct:>+9.1f}%", end="")
                    if key == "tpot":
                        print(f" {'':>10} {'':>10}")
                    elif key == "sp95":
                        print(f" {'':>10}")
                    else:
                        print()

    print(f"\n{'='*100}")
    print(f"  共 {len(rows)} 个运行，结果目录: {results_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
