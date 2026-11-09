"""Score this corpus on the two published protocols, and check the anchor.

Joins full-resolution RGB PSNR (3200x2432, 50 MAI validation frames) with the latency measured at
each paper's own runtime resolution, and applies each challenge's own normalisation constant:

  MIDD / MAI 2021 table   latency @ 720x480     C = 2^64          vs SplitterNet 37.92 dB / 27.7 ms
  MAI 2026 table          latency @ 1088x1920   C ~= 7.56e18      vs LiteDenoiseNet 37.58 dB / 46.1 ms

THE ANCHOR DECIDES WHETHER ANY OF IT COUNTS. LiteDenoiseNet is a reimplementation of the MAI 2026
student network trained on the same 650/50 split, and the paper reports 37.66 dB validation. If this
pipeline's re-measured full-resolution RGB number lands near that, the protocol is reproduced and
every other row is meaningful. If it does not, the gap is in OUR pipeline and no other row should be
quoted -- so it is checked first and printed loudest.

WHAT IS STILL NOT MATCHED, EVEN AFTER THIS RUN. Resolution and metric now agree with the papers;
SILICON does not. Ours is a Qualcomm Hexagon NPU. MIDD's 27.7 ms is an Exynos 2200 Mali GPU -- a
different vendor and a GPU rather than an NPU, and §22 measured a 53x spread between those two
execution paths, so the 720x480 latency column is NOT comparable to MIDD's and only the PSNR column
is. MAI 2026 is closer: its Snapdragon 8 Elite NPU column is the same vendor and engine class as
ours, so latency there is a fair comparison -- but its Final Score averages Snapdragon with a
Dimensity 9500 that we cannot measure, so a score built on our single-chip latency is not its score.
Both caveats are printed with the tables rather than left to the reader.

  python reexport_report.py
"""
import argparse
import json
import os

RX = "/shared/ssd/home/b-b-ahmed/thesis/sweep_out/reexport"
C_2021 = 2 ** 64
C_2026 = 7.56e18
ANCHOR = "LiteDenoiseNet"
ANCHOR_PUB = {"val_dB": 37.66, "test_dB": 37.58, "snap_npu_ms": 46.1, "dim_npu_ms": 34.0,
              "params": 1_960_000}
MIDD_PUB = [("Identity mapping", 30.88, None), ("ResNet 18", 37.48, 186.0), ("U-Net", 37.87, 95.0),
            ("Megvii", 38.05, 56.8), ("NOAHTCV", 37.81, 30.4), ("SplitterNet", 37.92, 27.7)]


def score(psnr, ms, C):
    return None if not (psnr and ms) else 2 ** (2 * psnr) / (C * ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RX)
    a = ap.parse_args()

    sub = json.load(open(os.path.join(a.dir, "subset.json")))
    order = [m["model"] for m in sub["models"]]

    psnr = {}
    p = os.path.join(a.dir, "fullres.jsonl")
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "OK":
                    psnr[r["model"]] = r
    lat = {}
    p = os.path.join(a.dir, "latency.json")
    if os.path.exists(p):
        for v in json.load(open(p)).values():
            lat.setdefault(v["model"], {})[(v["res"], "S26" if "S26" in v["device"] else "S24")] = v

    print(f"{len(psnr)} models with full-res RGB PSNR | {len(lat)} with re-exported latency")
    if not psnr:
        print("\nno full-res PSNR yet -- reexport-gpu has not run. Latency-only preview:")
        for m in order:
            for k, v in sorted(lat.get(m, {}).items()):
                print(f"  {m[:40]:<42} {k[0]:<10} {k[1]}  {v['ms']:8.2f} ms  "
                      f"NPU {v['npu_time_frac']*100:5.1f}%")
        return

    # ---- 1. the anchor -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. ANCHOR CHECK — LiteDenoiseNet against its published number")
    print("=" * 78)
    if ANCHOR in psnr:
        r = psnr[ANCHOR]
        got, pub = r["psnr_rgb_fullres"], ANCHOR_PUB["val_dB"]
        print(f"  published (distilled, 41.6M teacher) : {pub:.2f} dB")
        print(f"  ours (same graph, NO distillation)   : {got:.3f} dB   [{r.get('mode')}]")
        print(f"  shave-1px variant                    : {r['psnr_rgb_fullres_shave1']:.3f} dB")
        print(f"  per-frame range                      : {r['psnr_min']:.2f} .. {r['psnr_max']:.2f} dB")
        print(f"  DELTA                                : {got - pub:+.3f} dB")
        print("  Reading: the paper's whole contribution is high-alpha distillation from a 21x")
        print("  larger teacher, which this corpus does not do, so a negative delta of roughly")
        print("  one dB is the expected outcome and confirms the protocol rather than a defect.")
        print("  A POSITIVE delta, or a delta beyond a few dB, means the protocol is wrong.")
    else:
        print(f"  {ANCHOR} not evaluated yet -- nothing below should be quoted.")

    # ---- 2. MIDD / MAI 2021 axes ---------------------------------------------------
    for label, res, C, dev in (("2. MIDD / MAI 2021 AXES — latency @ 720x480, C = 2^64",
                                "480x720", C_2021, "S26"),
                               ("3. MAI 2026 AXES — latency @ 1088x1920, C ~= 7.56e18",
                                "1088x1920", C_2026, "S26")):
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        rows = []
        for m in order:
            pr = psnr.get(m)
            lv = lat.get(m, {}).get((res, dev))
            if not pr:
                continue
            rows.append((m, pr["psnr_rgb_fullres"], lv["ms"] if lv else None,
                         lv["npu_time_frac"] if lv else None,
                         score(pr["psnr_rgb_fullres"], lv["ms"] if lv else None, C)))
        rows.sort(key=lambda r: -(r[4] or -1))
        print(f"{'#':>3}  {'model':<40}{'RGB dB':>9}{'ms':>9}{'NPU':>7}{'score':>10}")
        for i, (m, p_, ms, nf, sc) in enumerate(rows, 1):
            print(f"{i:>3}  {m[:39]:<39}{p_:>9.3f}"
                  f"{(f'{ms:9.2f}' if ms else '        —')}"
                  f"{(f'{nf*100:6.0f}%' if nf is not None else '      —')}"
                  f"{(f'{sc:10.1f}' if sc else '         —')}")
        if res == "480x720":
            print("\n  published MIDD Table 2 for context (Exynos 2200 Mali GPU, NOT our silicon):")
            for n, p_, ms in MIDD_PUB:
                s = score(p_, ms, C_2021)
                print(f"      {n:<20}{p_:>8.2f} dB{(f'{ms:9.1f} ms' if ms else '        —')}"
                      f"{(f'{s:10.1f}' if s else '         —')}")
            print("  LATENCY IS NOT COMPARABLE HERE: theirs is a Mali GPU, ours a Hexagon NPU, and")
            print("  §22 measured a 53x spread between those execution paths. Compare the PSNR")
            print("  column, which now shares their protocol; treat the score as ours-only.")
        else:
            print(f"\n  published MAI 2026 for context: LiteDenoiseNet {ANCHOR_PUB['test_dB']} dB test,")
            print(f"      {ANCHOR_PUB['snap_npu_ms']} ms Snapdragon 8 Elite NPU / "
                  f"{ANCHOR_PUB['dim_npu_ms']} ms Dimensity 9500 NPU @ 1088x1920")
            print("  LATENCY IS COMPARABLE HERE (same vendor, same engine class), but their Final")
            print("  Score averages two chips and ours is Snapdragon-only, so the score column is")
            print("  ranked within this table and is not their leaderboard number.")

    out = {"anchor": {"model": ANCHOR, "published_val_dB": ANCHOR_PUB["val_dB"],
                      "measured": psnr.get(ANCHOR, {}).get("psnr_rgb_fullres")},
           "psnr_fullres_rgb": {m: v["psnr_rgb_fullres"] for m, v in psnr.items()},
           "latency": {m: {f"{k[0]}|{k[1]}": v["ms"] for k, v in d.items()} for m, d in lat.items()},
           "constants": {"C_2021": C_2021, "C_2026": C_2026}}
    o = os.path.join(a.dir, "reexport_report.json")
    json.dump(out, open(o, "w"), indent=1)
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
