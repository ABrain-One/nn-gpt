"""Build the ANONYMOUS artifact bundle for double-blind submission, from scratch, in one pass.

  python3 build_artifact.py              # build + zip to thesis/artifact/
  python3 build_artifact.py --out /tmp/x --no-zip
  python3 build_artifact.py --verify-only   # re-check an existing bundle's manifest

ANONYMITY IS THE HARD CONSTRAINT, AND IT IS ENFORCED TWICE. Every byte this script writes is
scanned before it lands, and the whole tree is swept again at the end. Either check aborts
the build. Do not weaken this to "the inputs were clean" -- the model source comes out of a
database whose contents are not under this script's control.

WHAT IS DELIBERATELY HELD BACK, AND WHY
  * trained weights      4,237 checkpoints, ~25 GB. Not needed: every quality figure the
                         paper reports is in metrics/.
  * random-walk source   the 14 DenoiseRW architectures. Their MEASUREMENTS stay, because the
                         "is search better than chance" comparison depends on them; only the
                         generated source is withheld.
  * the pipeline         withheld from the anonymous release only.
"""
import argparse, collections, hashlib, json, os, re, shutil, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pareto_metrics_v2 import producer, arch_id, canon, EPOCH_SUFFIX  # noqa: E402

T = "/shared/ssd/home/b-b-ahmed/thesis"
SW = f"{T}/sweep_out"
DEFAULT_OUT = f"{T}/artifact/lemur-denoise-artifact"
WITHHELD_SOURCE_PREFIX = "DenoiseRW"
BAN = re.compile(r"ahmed|kayani|abrain|b-b-ahmed|/shared/ssd|/a/mm|gmail|uni-w|wuerzburg|"
                 r"würzburg|julius|asadhm3d", re.I)
BAN_B = re.compile(rb"ahmed|kayani|abrain|b-b-ahmed|/shared/ssd|/a/mm|gmail|uni-w|"
                   rb"wuerzburg|w\xc3\xbcrzburg|julius|asadhm3d", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    OUT = a.out
    if a.verify_only:
        return verify(OUT)

    def wr(rel, text):
        p = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        m = BAN.search(text)
        if m:
            raise SystemExit("ANONYMITY ABORT: %s would leak %r" % (rel, m.group(0)))
        open(p, "w").write(text)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    fr = json.load(open(f"{SW}/fullres_union.json"))
    cr = json.load(open(f"{SW}/crop_psnr_union.json"))
    lat = json.load(open(f"{SW}/aihub/latency_corpus.json"))
    macj = json.load(open(f"{SW}/verify_macs.json"))
    mac, CAP = macj["macs"], float(macj["cap_MMac"])
    cen = json.load(open(f"{SW}/budget_census.json"))

    def idx(d, ok=lambda v: True):
        return {canon(EPOCH_SUFFIX.sub("", k)): v for k, v in d.items() if ok(v)}
    I_fr, I_cr = idx(fr), idx(cr)
    I_mac = idx(mac, lambda v: v is not None)
    L = collections.defaultdict(list)
    for k, v in lat.items():
        if isinstance(v, dict) and v.get("status") == "OK" and v.get("ms"):
            L[canon(EPOCH_SUFFIX.sub("", k))].append(v)

    con = sqlite3.connect(f"file:{T}/nn-dataset/db/ab.nn.db?mode=ro", uri=True)
    arch = {}
    for n, code in con.execute("select name, code from nn"):
        if code and arch_id(n) not in arch:
            arch[arch_id(n)] = (n, code)

    nsrc = 0
    for k, (n, code) in sorted(arch.items()):
        if n.startswith(WITHHELD_SOURCE_PREFIX):
            continue
        wr(f"models/{k[:2]}/{k}.py", code)
        nsrc += 1

    cols = ["arch_uid2", "name", "producer", "mmac", "edge_legal", "crop_psnr_db",
            "fullres_psnr_db", "epochs_reached", "latency_ms_s26u", "npu_time_frac_s26u",
            "n_layers_s26u", "latency_ms_s24", "in_paper_fullres", "in_paper_latency"]
    body, nlat = [], 0
    latrows = ["\t".join(["arch_uid2", "name", "device", "ms", "npu_time_frac",
                          "n_layers", "fallback_units"])]
    for k, (n, _) in sorted(arch.items()):
        cn = canon(EPOCH_SUFFIX.sub("", n))
        mm = I_mac.get(cn)
        recs = L.get(cn, [])
        s26 = next((v for v in recs if v.get("device") == "Samsung Galaxy S26 Ultra"), None)
        s24 = next((v for v in recs if v.get("device") == "Samsung Galaxy S24 (Family)"), None)
        body.append([k, n, producer(n),
                     str(round(float(mm), 1)) if mm is not None else "",
                     "" if mm is None else str(int(float(mm) <= CAP)),
                     str(I_cr.get(cn, "")), str(I_fr.get(cn, "")),
                     str(cen.get(n, {}).get("reach", "")),
                     str(s26["ms"]) if s26 else "",
                     str(s26.get("npu_time_frac", "")) if s26 else "",
                     str(s26.get("n_layers", "")) if s26 else "",
                     str(s24["ms"]) if s24 else "",
                     str(int(cn in I_fr)), str(int(bool(recs)))])
        for v in recs:
            latrows.append("\t".join([k, n, str(v.get("device", "")), str(v.get("ms", "")),
                                      str(v.get("npu_time_frac", "")), str(v.get("n_layers", "")),
                                      ",".join(v.get("fallback_units") or [])]))
            nlat += 1
    wr("metrics/corpus.tsv", "\n".join(["\t".join(cols)] + ["\t".join(r) for r in body]) + "\n")
    wr("metrics/latency_measurements.tsv", "\n".join(latrows) + "\n")

    ci = {c: i for i, c in enumerate(cols)}
    prod = collections.Counter(r[ci["producer"]] for r in body)
    n_fr = sum(1 for r in body if r[ci["in_paper_fullres"]] == "1")
    n_lt = sum(1 for r in body if r[ci["in_paper_latency"]] == "1")
    n_bo = sum(1 for r in body if r[ci["in_paper_fullres"]] == "1" and r[ci["in_paper_latency"]] == "1")
    n_lg = sum(1 for r in body if r[ci["edge_legal"]] == "1")
    n_50 = sum(1 for r in body if r[ci["epochs_reached"]] not in ("", "0")
               and int(r[ci["epochs_reached"]]) >= 50)

    gm = json.load(open(f"{HERE}/docs/deploy_gate_model.json"))
    wr("gate/deploy_gate_model.json", json.dumps(gm, indent=1))
    wr("gate/README.md",
       "# Frozen deployability gate\n\n"
       "`deploy_gate_model.json` holds the standardised logistic coefficients that predict full\n"
       "CPU eviction from source text alone. Features are abstract-syntax-tree counts and derived\n"
       "ratios; the profiler's own `n_layers` is deliberately NOT a feature, being a\n"
       "post-conversion quantity.\n\n"
       "Score a candidate as `z = b + sum_k w_k * (f_k - mu_k) / sd_k`, then\n"
       "`p_evict = 1 / (1 + exp(-z))`. The operating point used in the paper is `p >= 0.5`.\n\n"
       "## Features, in the order the coefficients are stored\n\n"
       + "\n".join("%2d. `%s`  (w=%+.4f)" % (i + 1, k, w)
                   for i, (k, w) in enumerate(zip(gm["keys"], gm["w"])))
       + "\n\nFitted on %d models (%d evicted).\n" % (gm["n_train"], gm["n_evicted"]))
    for f in ("pareto_v3_matched50_s26u.json", "pareto_v3_bootstrap.json",
              "matched50_crop_psnr.json", "budget_sensitivity.json"):
        wr(f"pareto/{f}", open(f"{SW}/{f}").read())

    wr("README.md", README % dict(
        nsrc="{:,}".format(nsrc), narch="{:,}".format(len(body)), cap=int(CAP),
        n_fr=n_fr, n_lt=n_lt, n_bo=n_bo, n_lg=n_lg, n_50=n_50, nlat=nlat,
        nrw=len(body) - nsrc,
        byprod="\n".join("- %s: %d" % (k, v) for k, v in prod.most_common())))
    wr("LICENSE.md", LICENSE)

    files = manifest_files(OUT)
    json.dump({"artifact": "edge-denoising architecture corpus (anonymous)",
               "n_files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()),
               "n_architectures": len(body), "n_source_files": nsrc,
               "counts": {"full_res": n_fr, "latency": n_lt, "both": n_bo,
                          "edge_legal": n_lg, "trained_50_epochs": n_50},
               "by_producer": dict(prod), "files": files},
              open(f"{OUT}/MANIFEST.json", "w"), indent=1)

    bad = sweep(OUT)
    if bad:
        raise SystemExit("ANONYMITY SWEEP FAILED: %s" % bad[:5])
    print("built %s: %d files, %d source, %d architectures, %d latency rows"
          % (OUT, len(files) + 1, nsrc, len(body), nlat))
    print("anonymity sweep: CLEAN over %d files" % (len(files) + 1))

    if not a.no_zip:
        base = os.path.basename(OUT)
        z = os.path.join(os.path.dirname(OUT), base + ".zip")
        if os.path.exists(z):
            os.remove(z)
        root = os.path.dirname(OUT)
        os.system("cd %s && zip -q -r %s.zip %s" % (root, base, base))
        print("zip: %s (%.1f MB)" % (z, os.path.getsize(z) / 1e6))


def manifest_files(OUT):
    files = {}
    for root, _, fs in os.walk(OUT):
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, OUT)
            if rel == "MANIFEST.json":
                continue
            b = open(p, "rb").read()
            files[rel] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    return files


def sweep(OUT):
    bad = []
    for root, _, fs in os.walk(OUT):
        for f in fs:
            p = os.path.join(root, f)
            m = BAN_B.search(open(p, "rb").read())
            if m:
                bad.append((os.path.relpath(p, OUT), m.group(0).decode("utf8", "replace")))
    return bad


def verify(OUT):
    m = json.load(open(os.path.join(OUT, "MANIFEST.json")))
    bad = [f for f, v in m["files"].items()
           if hashlib.sha256(open(os.path.join(OUT, f), "rb").read()).hexdigest() != v["sha256"]]
    print("MISMATCH: %s" % bad[:5] if bad else "all %d files verify" % len(m["files"]))
    s = sweep(OUT)
    print("anonymity sweep: %s" % ("CLEAN" if not s else "LEAKS %s" % s[:5]))
    return 1 if (bad or s) else 0


README = r"""# Edge-denoising architecture corpus --- anonymous artifact

Supporting artifact for a double-blind conference submission. It contains the architecture
corpus, every measurement the paper reports, and the frozen deployability gate. It contains
no author, institution or repository identifier.

## Contents

| path | what it is |
|---|---|
| `models/<xx>/<arch_uid2>.py` | model source, sharded by the first two hex characters of `arch_uid2`. %(nsrc)s files for %(narch)s architectures: the %(nrw)d random-walk-control sources are held back (see *What is not included*) |
| `metrics/corpus.tsv` | one row per architecture: producer, MACs, both quality axes, epochs reached, flagship latency and residency |
| `metrics/latency_measurements.tsv` | one row per (architecture, device) measurement, all devices |
| `gate/deploy_gate_model.json` | frozen logistic coefficients of the source-level eviction gate |
| `gate/README.md` | how to score a candidate, and the feature list in coefficient order |
| `pareto/` | the matched-budget Pareto front, the size-matched bootstrap, the matched-50 quality axis, and the budget-sensitivity table |

## Counts

| quantity | value |
|---|---|
| distinct architectures (by `arch_uid2`) | **%(narch)s** |
| with a full-resolution score | %(n_fr)d |
| with at least one on-device latency measurement | %(n_lt)d |
| with both quality and measured latency | %(n_bo)d |
| edge-legal (at or under the %(cap)d MMac cap) | %(n_lg)d |
| trained to a full 50 epochs | %(n_50)d |
| per-(architecture, device) latency measurements | %(nlat)d |

By producer:

%(byprod)s

### Why this is larger than the count in the paper

The paper reports **4,276** architecturally distinct networks. That is the figure from the
distinctness audit run at submission time, over 4,529 database rows with 253 exact duplicates
removed. The database accumulated further architectures afterwards, so this release carries
%(narch)s. Nothing was removed; the release is a superset. `in_paper_fullres` and
`in_paper_latency` in `corpus.tsv` let you reconstruct the measured subsets exactly.

## Two quality axes, and a caveat that matters if you reuse them

`crop_psnr_db` is Y-channel PSNR on a 256x256 centre crop --- the cheap screening metric the
search selected on. `fullres_psnr_db` is RGB PSNR over 50 full-resolution frames at
3200x2432, the published evaluation protocol. They agree closely on the *same weights* but
**do not agree in rank**, and the paper's own proxy table scores the crop metric as "coarse
only" against the full-resolution one.

**Known defect, stated rather than hidden.** The crop column is a screening table and most of
its entries are 5-epoch scores, but for 41 of 58 hand-designed architectures it holds their
50-epoch value instead, against 9 of 476 generated architectures. Any per-producer comparison
drawn from `crop_psnr_db` alone is therefore confounded with training budget. Use
`epochs_reached` to restrict to a matched budget; `pareto/matched50_crop_psnr.json` is the
matched-50 axis the paper's Pareto appendix uses.

## Latency

Measurements come from a commercial device farm. `metrics/latency_measurements.tsv` records
the device for every row, and the device must be treated as a covariate rather than
normalised away: the absolute offset between handset generations is a median 1.59x even
though the rank correlation between them is 0.987.

`npu_time_frac` is the fraction of execution time on the neural accelerator. `1.0` is fully
resident, `0.0` is full eviction to the CPU --- the label the gate predicts.

## What is not included

- **Trained weights.** 4,237 checkpoints totalling about 25 GB, too large for this release.
  They are not needed to reproduce any table here, since every reported quality figure is in
  `metrics/`.
- **Source for the random-walk control arm** (%(nrw)d architectures, producer
  `random walk (control)`). Its *measurements* are included and flagged by producer, because
  the paper's "is search better than chance" comparison depends on them; only the generated
  source files are held back. Every number the paper reports about that arm is reproducible
  from `metrics/`.
- **The generation and training pipeline.** Withheld from the anonymous release only; it
  accompanies the de-anonymised version.

## Integrity

`MANIFEST.json` lists every file with its SHA-256 and byte count. Verify with:

```
python3 -c "import json,hashlib,os; m=json.load(open('MANIFEST.json'));
bad=[f for f,v in m['files'].items() if hashlib.sha256(open(f,'rb').read()).hexdigest()!=v['sha256']];
print('MISMATCH:',bad) if bad else print('all',len(m['files']),'files verify')"
```

## Licence

Model source and measurement tables are released under CC BY 4.0. See `LICENSE.md`.
"""

LICENSE = """# Licence

## Measurement tables, Pareto artifacts and the frozen gate
Creative Commons Attribution 4.0 International (CC BY 4.0).
You may share and adapt them for any purpose, including commercially, provided you give
appropriate credit. Full text: https://creativecommons.org/licenses/by/4.0/

## Model source files under `models/`
Also CC BY 4.0. These files were produced by automated search and by language-model
generation, and are released as data accompanying a scientific publication.

## No warranty
Provided as is, without warranty of any kind. The measurements were taken on a commercial
device farm at a particular time with a particular compiler toolchain; the paper documents
that such measurements drift between toolchain versions, so they should be read as a dated
record and not as a specification of any device's performance.
"""

if __name__ == "__main__":
    raise SystemExit(main())
