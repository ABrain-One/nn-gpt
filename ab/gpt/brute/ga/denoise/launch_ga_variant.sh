#!/usr/bin/env bash
# PREDICTOR DISABLED 2026-08-07. The in-house zero-cost surrogate ranks ANTI-correlated with
# truth: spearman -0.178 over 326 production models (predictor_tracking_history.csv), against
# the +0.693 its own k-fold CV claimed. With --predict-keep 0.5 it trained the top half of every
# generation by that ranking, i.e. it preferentially discarded the BETTER half. Use epoch-3 PSNR
# for selection instead (spearman +0.728, 82% top-half overlap) -- see predict_final.py.
# One of four INDEPENDENT GA populations, parameterised by $VARIANT.
#
# WHY FOUR SEPARATE RUNS rather than sharding one generation across four GPUs. Sharding
# within a generation needs each worker to own a LEMUR root and DB copy, because the root is
# resolved from CWD *at import time* and ab.nn is already loaded in the parent. That forces
# spawn-based workers plus a refactor splitting "train one model" out of make_fitness, which
# also owns the fitness cache, the diversity gate and the edge-cost folding -- surgery on the
# professor's GA engine with a real chance of quietly altering evolutionary semantics.
#
# Four independent populations get the same ~4x throughput with no code change at all, and
# for a corpus intended as FINE-TUNING DATA they are arguably the better experiment: four
# populations explore four regions and cannot collapse onto one another, whereas one deeper
# search converges. The cost is depth -- no single run goes further than 30 generations.
#
# The variants deliberately differ on the axes that change WHICH ARCHITECTURES get explored
# (seed pool, mutation weights, cost pressure), not on cosmetic knobs, so the union is a
# wider corpus rather than four samples of the same distribution.
set -euo pipefail
V_EARLY="${VARIANT:?set VARIANT=cheap|struct|graded}"

# EACH VARIANT OWNS ITS OWN DATABASE FILE. The existing normv3 run writes
# /a/mm/nn-dataset/db/ab.nn.db directly; pointing three more runs at that same file would put
# four concurrent writers on an NFS-hosted SQLite database, which interleaves and corrupts
# rather than blocking -- the exact failure the LLM sharding was built to avoid, and this DB
# is the only copy of ~800 models.
#
# So each run gets a private root: ab/ and README.md symlinked (LEMUR resolves its root from
# CWD by looking for both), data symlinked, and db/ pointing at a per-variant copy. The copy
# lives on SHARED storage, not the pod-local emptyDir, so a 30-hour run's results survive the
# pod. One writer per file means no contention at all.
#
# Results are folded back with merge_shard_dbs.py when the run finishes -- single writer,
# insert-only, dry-run first.
GADB=/a/mm/sweep_out/ga_db/$V_EARLY
ROOT=/scratch/ga_$V_EARLY
mkdir -p "$GADB" "$ROOT"
for l in ab README.md data train validation; do
  [ -e "/a/mm/nn-dataset/$l" ] && ln -sfn "/a/mm/nn-dataset/$l" "$ROOT/$l"
done
if [ ! -f "$GADB/ab.nn.db" ]; then
  echo "seeding private DB for $V_EARLY (online snapshot of the live master)"
  python -c "import sqlite3,sys; a=sqlite3.connect(sys.argv[1]); b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close(); a.close()" \
      /a/mm/nn-dataset/db/ab.nn.db "$GADB/ab.nn.db"
fi
ln -sfn "$GADB" "$ROOT/db"

cd "$ROOT"
[ -d "$ROOT/ab" ] && [ -f "$ROOT/README.md" ] || { echo "FATAL: private root not formed"; exit 1; }
export PYTHONPATH="$ROOT"
export BRUTE_NORM_SAFE=1
export BRUTE_DEBUG=0
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
export LEMUR_CKPT_DIR="${LEMUR_CKPT_DIR:-/a/mm/sweep_out/ckpt}"
NN=/a/mm/nn-dataset/ab/nn/nn
# 9053 MMac is the generic "edge-legal" budget. The MAI Full HD latency target needs a far
# tighter one: our measured fit fullhd_ms = 0.0082*MMac + 22.87 (r=0.949, n=24, every model
# npu_layer_frac=1.0 so nothing is evicted and compute dominates) puts 46.1 ms at 2820 MMac.
MAC_CAP="${MAC_CAP:-9053}"
POP="${POP:-50}"; GENS="${GENS:-30}"; EP="${EP:-5}"
# BENCHMARK. Default is MAI's own pairs, where a no-op scores 33.02 dB and the best of 1,462
# models scores 35.15 -- about 4 dB of total room, which is why 73% of the corpus sits inside
# a single dB and nothing separates. denoise_awgn re-noises the SAME clean images at a chosen
# sigma; at 25 the headroom is ~9 dB. Scores across the two are NOT comparable, and the
# separate dataset key in `stat` is what prevents them being mixed.
DATASET="${DATASET:-denoise}"
V="$V_EARLY"

case "$V" in
  # ---- CHEAP: the low-MAC regime, which is where published mobile denoisers actually live.
  # Our whole corpus sits 5-20x above MFDNet-L; nothing has deliberately searched down there.
  # Seeded from the six cheapest legal families (913-1479 MMac, all measured) and pushed with
  # a heavy cost weight so MACs are penalised hard rather than treated as a pass/fail cap.
  cheap)
    SEEDS_DB="DenoiseLapPyr DenoiseParallelMR DenoiseThunder DenoiseMFDNet DenoiseSplitterNet DenoiseUnet5F"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.25,activation:0.10,layer_type:0.20,kernel_size:0.15,stride:0.10,scale:0.15,skip:0.05"
    EXTRA="--cost-weight 0.8 --cost-mode log --skip-prob 0.2"
    ;;
  # ---- STRUCT: topology rather than width. Seeds chosen for DISTINCT structural motifs
  # (attention, inverted bottleneck, channel shuffle, split-merge, sub-pixel, pool-deconv) and
  # mutation weighted onto layer_type/skip so the search rewires graphs instead of resizing
  # them. This is the arm most likely to produce architectures unlike anything hand-written.
  struct)
    SEEDS_DB="DenoiseSelfAttn DenoiseMBConv DenoiseChanShuffle DenoiseSplitterNetF DenoiseSubPixel DenoisePoolDeconv"
    SEEDS_FILES="$NN/DenoiseRestormerLite.py $NN/DenoiseNAFNet.py $NN/DenoiseMegvii.py"
    WEIGHTS="dimension:0.05,activation:0.10,layer_type:0.40,kernel_size:0.10,stride:0.05,scale:0.10,skip:0.20"
    EXTRA="--cost-weight 0.2 --cost-mode log --skip-prob 0.5"
    ;;
  # ---- GRADED: the diversity gate has been EXACT-HASH ONLY in every GA run to date. For a
  # mutation-only generator that is defensible (mutants cluster >=0.985 by construction), but
  # this run is crossover-heavy, where exact-hash uniqueness certifies near-duplicates as
  # distinct. Turning graded on tests whether it changes the population -- an open question
  # nobody has measured, and cheap to answer with a whole run.
  graded)
    SEEDS_DB="DenoiseLPPool DenoiseMBConv DenoiseELUCtx DenoiseLinearMix DenoiseDilatedFlat DenoiseReflectPad"
    SEEDS_FILES="$NN/DenoiseResBN.py $NN/DenoiseLayerNorm.py $NN/DenoiseInstNorm.py"
    WEIGHTS="dimension:0.10,activation:0.25,layer_type:0.20,kernel_size:0.20,stride:0.10,scale:0.10,skip:0.05"
    EXTRA="--diversity-graded --cost-weight 0.3 --cost-mode log --skip-prob 0.35"
    ;;
  # ---- ASTUID: the DISTANT configuration with the identity swapped underneath it.
  # Exact identity comes from the AST of every method (prof/uid_full.py) instead of an fx
  # trace of forward. Measured on 1,468 models: 1,411 distinct vs 1,348, 100% coverage vs
  # 97.5%, and 38 groups separated whose forward graphs match but whose optimizer, loss or
  # update loop differ. Because the hash is FINER, this gate rejects FEWER candidates as
  # clones -- two models that share a forward graph but train differently are now both kept,
  # which is correct: they reach different accuracies, so they are different data points.
  #
  # Store is fresh by construction (ga_astuid_checkpoint_wlhashes.pt does not exist). Mixing
  # hash spaces in one store does not fail, it silently behaves as an empty reference set.
  astuid)
    SEEDS_DB="DenoiseUnet5 DnSeedGA-2d592837ae3fafe363b149f898878ff1 \
DnSeedGA-58ea50814ce92855e16864df642cf58f \
LLM_Denoise_cold-77ae6899fd1a1e3855044a7f4c8a2573 DenoiseSplitterNetF \
DenoiseGA-3907049bdf88b626f69e95c7ef019f8e DenoiseUnet3 \
DenoiseGA-e91a31ea12f50f71fd9b2a6c0c6eb8bd DenoiseGA-a2ed723e805542155bd8070791210708 \
LLM_Pg15_seeded-8427175997a3bbb1ebfb564cfb2da2f4 DenoiseUnet6 DenoiseUnet9F"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.05,activation:0.05,layer_type:0.35,kernel_size:0.10,stride:0.05,scale:0.15,skip:0.25"
    EXTRA="--cost-weight 0.3 --cost-mode log --skip-prob 0.5 --mutation-rate 0.6 --ast-exact"
    ;;
  # ---- AWGN25: the DISTANT configuration on a benchmark with room in it. Same seeds, same
  # operators, same gate -- the only thing that changes is the dataset, so any difference in
  # how far apart the resulting models land is attributable to the benchmark and not to the
  # search. That is the whole experiment: do architectures spread over 5+ dB at sigma 25
  # where they spread over 1 dB now? If they do, the metric was the binding constraint all
  # along. If they still cluster, the limit is the architectures and no benchmark fixes it.
  awgn25)
    SEEDS_DB="DenoiseUnet5 DnSeedGA-2d592837ae3fafe363b149f898878ff1 \
DnSeedGA-58ea50814ce92855e16864df642cf58f \
LLM_Denoise_cold-77ae6899fd1a1e3855044a7f4c8a2573 DenoiseSplitterNetF \
DenoiseGA-3907049bdf88b626f69e95c7ef019f8e DenoiseUnet3 \
DenoiseGA-e91a31ea12f50f71fd9b2a6c0c6eb8bd DenoiseGA-a2ed723e805542155bd8070791210708 \
LLM_Pg15_seeded-8427175997a3bbb1ebfb564cfb2da2f4 DenoiseUnet6 DenoiseUnet9F"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.05,activation:0.05,layer_type:0.35,kernel_size:0.10,stride:0.05,scale:0.15,skip:0.25"
    EXTRA="--cost-weight 0.3 --cost-mode log --skip-prob 0.5 --mutation-rate 0.6"
    ;;
  # ---- DISTANT: seeds chosen by MEASUREMENT, not by hand. Every other variant's seed list
  # was picked on a plausible story ("the cheapest legal ones", "distinct structural motifs")
  # that nobody checked. Measured, the hand-picked struct set has a pair at cosine 0.835 --
  # inside near-duplicate territory. These twelve come from farthest-point sampling over all
  # 950 models above 33.5 dB: no two exceed 0.759, below the 0.752 cross-family median, so
  # they are genuinely different families by construction rather than by assertion. The set
  # also spans all three arms (hand-written, GA, LLM), which no hand-picked list did.
  #
  # CROSSOVER-HEAVY, and this is the point of pairing it with distant seeds. The decision
  # ledger measured that mutation NEVER escapes its family (mutants cluster >=0.985) while
  # cross-family block crossover DOES (11/20 pairs below 0.95). Twelve distant families plus
  # an operator that can actually combine them is the only configuration where the GA can
  # reach architectures none of its seeds already occupy.
  #
  # Gate stays EXACT-HASH. The graded gate at 0.95 took the `graded` variant from ~17
  # models/gen to 1.6, and at the LLM's new 0.90 it would reject 95% of mutants outright
  # (same-family p05 = 0.926). Diversity here comes from the seeds and the operator, not
  # from filtering harder.
  distant)
    SEEDS_DB="DenoiseUnet5 DnSeedGA-2d592837ae3fafe363b149f898878ff1 \
DnSeedGA-58ea50814ce92855e16864df642cf58f \
LLM_Denoise_cold-77ae6899fd1a1e3855044a7f4c8a2573 DenoiseSplitterNetF \
DenoiseGA-3907049bdf88b626f69e95c7ef019f8e DenoiseUnet3 \
DenoiseGA-e91a31ea12f50f71fd9b2a6c0c6eb8bd DenoiseGA-a2ed723e805542155bd8070791210708 \
LLM_Pg15_seeded-8427175997a3bbb1ebfb564cfb2da2f4 DenoiseUnet6 DenoiseUnet9F"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.05,activation:0.05,layer_type:0.35,kernel_size:0.10,stride:0.05,scale:0.15,skip:0.25"
    EXTRA="--cost-weight 0.3 --cost-mode log --skip-prob 0.5 --mutation-rate 0.6"
    ;;
  # ---- NEWSEEDS: ONLY the 19 published-architecture families written and 50-epoch-measured
  # for this corpus (Aug 2026), spanning MPRNet/CBDNet/MemNet/FFDNet/MWCNN/DPIR/BRDNet/HINet/
  # RDN/KPN/axial-attention/MoE/FFC/NBNet/N3Net/SRN/InvDN/IRNN motifs. Distant-style config:
  # crossover-heavy with mutation weighted onto layer_type/skip, because these families are
  # distinct by construction (vet cos 0.84-0.95 vs corpus) and block crossover is the one
  # operator measured to escape a seed's family. Gate is AST-exact (--ast-exact) rather than
  # fx: DenoiseFourier's complex FFT is fx-untraceable, and the AST identity covers all 19.
  newseeds)
    SEEDS_DB="DenoiseMultiStage DenoiseNoiseEst DenoiseRecMem DenoiseUnshuffle \
DenoiseWaveletMS DenoiseUnfold DenoiseDualBranch DenoiseHalfInstNorm DenoiseResDense \
DenoiseKPN DenoiseKernelMix DenoiseAxial DenoiseMoE DenoiseFourier DenoiseLowRank \
DenoiseNonLocal DenoiseScaleRec DenoiseCoupling DenoiseFourDir"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.05,activation:0.05,layer_type:0.35,kernel_size:0.10,stride:0.05,scale:0.15,skip:0.25"
    EXTRA="--cost-weight 0.3 --cost-mode log --skip-prob 0.5 --mutation-rate 0.6 --ast-exact"
    ;;
  # ---- NS-ACT: the same 19 founders, mutation weighted onto ACTIVATIONS and KERNELS
  # instead of topology. Differentiated from `newseeds` on the axis that changes which
  # architectures get explored: that arm rewires graphs, this one retunes their local
  # operators. Same AST-exact gate (DenoiseFourier is fx-untraceable).
  ns-act)
    SEEDS_DB="DenoiseMultiStage DenoiseNoiseEst DenoiseRecMem DenoiseUnshuffle \
DenoiseWaveletMS DenoiseUnfold DenoiseDualBranch DenoiseHalfInstNorm DenoiseResDense \
DenoiseKPN DenoiseKernelMix DenoiseAxial DenoiseMoE DenoiseFourier DenoiseLowRank \
DenoiseNonLocal DenoiseScaleRec DenoiseCoupling DenoiseFourDir"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.10,activation:0.30,layer_type:0.10,kernel_size:0.25,stride:0.05,scale:0.15,skip:0.05"
    EXTRA="--cost-weight 0.2 --cost-mode log --skip-prob 0.3 --ast-exact"
    ;;
  # ---- NS-CHEAP: the same 19 founders under HEAVY cost pressure, pushing the new motifs
  # down into the low-MAC regime where published mobile denoisers live -- the axis the
  # original `cheap` variant searched, but from a founding pool it never had (its seeds
  # were the six cheapest legacy families; these are 19 measured published motifs).
  # ---- LATCAP: the first arm searched at ISO-COMPUTE with the published baseline rather than
  # under the generic edge cap. MAC_CAP=3530 is LiteDenoiseNet's OWN measured cost
  # (3,529,506,816 MACs) -- the model whose published 37.66 dB @ 46.1 ms we are comparing
  # against -- so no latency regression, intercept estimate or extrapolation enters the cap at
  # all. The existing corpus was built at 9053 MMac, 2.56x this budget, which is why our best
  # models are 1.6-2.1x too slow. An earlier version of this variant used 2820 MMac, derived by
  # applying the PUBLISHED 46.1 ms to OUR instrument's scale, which already reads +8.6% high;
  # that made the cap ~17% tighter than the real budget. Seeded from BOTH sides of the budget:
  # the five sub-1000 MMac DenoiseGA models (305-904 MMac), which have room to grow INTO the
  # budget, and the four fastest profiled DnSeedGA/Lite models (2926-3530 MMac), which sit just
  # over it and need to shrink. Growing into a cap is easier for the mutator than shrinking out
  # of one, so the small founders matter more than the large ones.
  latcap)
    SEEDS_DB="LiteDenoiseNet \
DenoiseGA-1eb5d8bb77883d58c25fe9d5fe7cdf49 DenoiseGA-c9c641143a88fa89e7e3de20a00cb283 \
DenoiseGA-51622f51571ca812a3013bb5609b20fc DenoiseGA-f48739d40278829171b9eea8c7df3433 \
DenoiseGA-c7fb9b80c8e0eaec339f834a0bb48b07 \
DnSeedGA-e230a3b4969a3fe867be85d00df3786b DnSeedGA-cc887dcdd6582160e9e202ee5b557933 \
DnSeedGA-67d7a72a608765e457ed4611322f94fb"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.25,activation:0.10,layer_type:0.20,kernel_size:0.15,stride:0.10,scale:0.15,skip:0.05"
    # cost-budget pinned to the cap itself with a one-sided penalty: inside the budget the GA
    # optimises PSNR alone (the hard --mac-cap already enforces legality), so no fitness is
    # spent trading dB for MACs we are allowed to use.
    EXTRA="--cost-weight 0.3 --cost-mode soft --cost-budget 3529506816 --skip-prob 0.35 --mutation-rate 0.6"
    ;;
  ns-cheap)
    SEEDS_DB="DenoiseMultiStage DenoiseNoiseEst DenoiseRecMem DenoiseUnshuffle \
DenoiseWaveletMS DenoiseUnfold DenoiseDualBranch DenoiseHalfInstNorm DenoiseResDense \
DenoiseKPN DenoiseKernelMix DenoiseAxial DenoiseMoE DenoiseFourier DenoiseLowRank \
DenoiseNonLocal DenoiseScaleRec DenoiseCoupling DenoiseFourDir"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.25,activation:0.10,layer_type:0.20,kernel_size:0.15,stride:0.10,scale:0.15,skip:0.05"
    EXTRA="--cost-weight 0.8 --cost-mode log --skip-prob 0.2 --ast-exact"
    ;;
  # ---- DNSEED2: the second DnSeedGA run, and the first seeded on MEASURED HARDWARE LATENCY
  # rather than PSNR alone. Founders are the 8 best points of the PSNR-vs-latency Pareto
  # frontier over the 202 DnSeedGA models that have both a PSNR and a Galaxy S26 Ultra
  # measurement (34.020-34.761 dB at 1.06-2.22 ms). Every previous arm seeded on dB and hoped
  # the result was deployable; §21 measured that hope failing -- 26% of MAC-legal models exceed
  # 10 ms, and the DenoiseGA cluster sits at a 47 ms median. Starting from the frontier is the
  # cheapest available correction, because it costs nothing and the frontier already exists.
  #
  # WHY DnSeedGA AND NOT DenoiseGA FOUNDERS. DnSeedGA is the strongest arm on every axis
  # measured: best real-noise transfer, 87% above the no-op against 26% for the LLM arm, and a
  # 1.53 ms median against DenoiseGA's 47.49 ms at n=597. It is also the arm the reference-bias
  # A/B (§31) was built around, so improving it is the intervention with the most context.
  #
  # Config follows `distant` (crossover-heavy, topology-weighted) because block crossover is the
  # one operator measured to escape a seed's family, and escaping the family is the whole point
  # of a second run from the same lineage -- otherwise this just resamples DnSeedGA's
  # neighbourhood. Cost weight 0.4 sits between `distant`'s 0.3 and `cheap`'s 0.8: the founders
  # are already fast, so heavy MAC pressure would buy latency that is already paid for and spend
  # the search's budget on shrinking instead of exploring.
  #
  # The no-op export gate (--noop-floor, default on) is new as of this arm: 19.1% of the first
  # DnSeedGA run's 376 models measured below no-op + noise and should never have reached the
  # corpus. Those children are still scored and still steer selection; they just are not exported.
  dnseed3)
    # CONTROLLED REPLICATION OF dnseed2: identical seeds and identical mutation weights, so the
    # ONLY changed variables are the fitness metric and the export gate. dnseed2's lineage went
    # 36.87 -> 34.89 full-res while its crop fitness went UP by 0.118 dB; this tests whether
    # scoring on the published metric stops that.
    #
    # THE EXPORT BAR IS 35.0 dB FULL-RES, chosen from the corpus shape rather than taste: of 2,945
    # measured models 91% sit in 33.0-34.5 and only 33 clear 35.0. Anything admitted here lands in
    # a region the corpus barely covers. Rejected models still score and still steer the search --
    # this gates the CORPUS, not the exploration.
    SEEDS_DB="DnSeedGA-20bab3230791d09509d138a198760e60 \
DnSeedGA-1a1a00ad2b0643217f2850f30d5b6f15 \
DnSeedGA-1b8735ac4692a071042d9bad7be47e53 \
DnSeedGA-61eec9f0e77e5e886865bc6e9d2d847f \
DnSeedGA-40df18220ca7be1f6215a0decf714743 \
DnSeedGA-b10bb5f7a6fb67d4bb98ba8f950c9fcb \
DnSeedGA-135a8efda24cdb3268c25472bb84b319 \
DnSeedGA-02d4928faae3980e4ba69359e9543255"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.10,activation:0.05,layer_type:0.30,kernel_size:0.10,stride:0.05,scale:0.15,skip:0.25"
    EXTRA="--cost-weight 0.4 --cost-mode log --skip-prob 0.5 --mutation-rate 0.6 --ast-exact --prefix DnSeed3 \
--fullres-fitness --deploy-gate --deploy-max-ops 203 --noop-floor 35.0 --noop-margin 0"
    ;;
  dnseed2)
    SEEDS_DB="DnSeedGA-20bab3230791d09509d138a198760e60 \
DnSeedGA-1a1a00ad2b0643217f2850f30d5b6f15 \
DnSeedGA-1b8735ac4692a071042d9bad7be47e53 \
DnSeedGA-61eec9f0e77e5e886865bc6e9d2d847f \
DnSeedGA-40df18220ca7be1f6215a0decf714743 \
DnSeedGA-b10bb5f7a6fb67d4bb98ba8f950c9fcb \
DnSeedGA-135a8efda24cdb3268c25472bb84b319 \
DnSeedGA-02d4928faae3980e4ba69359e9543255"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.10,activation:0.05,layer_type:0.30,kernel_size:0.10,stride:0.05,scale:0.15,skip:0.25"
    EXTRA="--cost-weight 0.4 --cost-mode log --skip-prob 0.5 --mutation-rate 0.6 --ast-exact --prefix DnSeedGA2"
    ;;
  # ---- DNSEED2-ACT: the SAME frontier founders as dnseed2, mutation weighted onto ACTIVATIONS
  # and KERNELS instead of topology. Differentiated on the axis that changes which architectures
  # get explored: dnseed2 rewires graphs, this arm retunes their local operators and leaves the
  # wiring alone. That distinction matters more than usual here, because §28 measured topology
  # edits to be worth only 0.307 dB across the whole 2x2 while §30 measured 1.188 dB from how a
  # single connection is processed -- so local-operator search is not the lesser axis it looks like.
  dnseed2-act)
    SEEDS_DB="DnSeedGA-20bab3230791d09509d138a198760e60 \
DnSeedGA-1a1a00ad2b0643217f2850f30d5b6f15 \
DnSeedGA-1b8735ac4692a071042d9bad7be47e53 \
DnSeedGA-61eec9f0e77e5e886865bc6e9d2d847f \
DnSeedGA-40df18220ca7be1f6215a0decf714743 \
DnSeedGA-b10bb5f7a6fb67d4bb98ba8f950c9fcb \
DnSeedGA-135a8efda24cdb3268c25472bb84b319 \
DnSeedGA-02d4928faae3980e4ba69359e9543255"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.10,activation:0.30,layer_type:0.10,kernel_size:0.25,stride:0.05,scale:0.15,skip:0.05"
    EXTRA="--cost-weight 0.3 --cost-mode log --skip-prob 0.3 --ast-exact --prefix DnSeed2Act"
    ;;
  # ---- DNSEED2-FAST: founders from the FAST end of the frontier (0.70-1.06 ms, 33.311-34.020 dB)
  # under heavy cost pressure, pushing for the sub-millisecond regime. This is the arm aimed at
  # the MAI-style objective, where runtime sits in the denominator and a 2x speedup outweighs a
  # few tenths of a dB.
  #
  # The 0.54 ms frontier point is DELIBERATELY NOT a founder: it measures 32.893 dB, below the
  # 33.021 no-op. It is Pareto-optimal only because nothing faster exists, and seeding a search
  # from a model that does not denoise would spend 30 generations refining a no-op.
  dnseed2-fast)
    SEEDS_DB="DnSeedGA-e230a3b4969a3fe867be85d00df3786b \
DnSeedGA-adfa1d6bc5a85d701ae49f572642f562 \
DnSeedGA-500fa835825df1c6977b7976ec2cc253 \
DnSeedGA-cc887dcdd6582160e9e202ee5b557933 \
DnSeedGA-0b1bc4aeac1ea3730f087b846df4d221 \
DnSeedGA-02d4928faae3980e4ba69359e9543255"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.25,activation:0.10,layer_type:0.20,kernel_size:0.15,stride:0.10,scale:0.15,skip:0.05"
    EXTRA="--cost-weight 0.8 --cost-mode log --skip-prob 0.2 --ast-exact --prefix DnSeed2Fast"
    ;;
  # ---- DNSEED2-WIDE: all 13 above-bar frontier points as founders rather than the top 8, with
  # mutation weighted onto DIMENSION. The other three arms start from a narrow, high-quality
  # cluster; this one starts from the whole frontier (0.70-2.22 ms, 33.311-34.761 dB) so the
  # population spans the tradeoff curve instead of one end of it, and searches width/depth.
  #
  # Low cost weight on purpose: dimension mutation already moves MACs, so adding heavy cost
  # pressure on the same axis would just make the arm a slower duplicate of dnseed2-fast.
  dnseed2-wide)
    SEEDS_DB="$(cat /a/mm/sweep_out/dnseed2_all_seeds.txt)"
    SEEDS_FILES=""
    WEIGHTS="dimension:0.35,activation:0.10,layer_type:0.20,kernel_size:0.15,stride:0.05,scale:0.10,skip:0.05"
    EXTRA="--cost-weight 0.2 --cost-mode log --skip-prob 0.3 --mutation-rate 0.6 --ast-exact --prefix DnSeed2Wide"
    ;;
  *) echo "unknown VARIANT=$V"; exit 2 ;;
esac
export BRUTE_WEIGHTS="$WEIGHTS"

echo "=== GA VARIANT=$V pop=$POP gens=$GENS ep=$EP  $(date -Is) ==="
echo "db seeds   : $SEEDS_DB"
echo "file seeds : ${SEEDS_FILES:-<none>}"
echo "weights    : $WEIGHTS"
echo "extra      : $EXTRA"

# Every path is namespaced by variant. Sharing a checkpoint, cache or models-dir between
# concurrent runs would have them overwrite each other's population and silently merge four
# searches into one corrupted one.
python -u "$D/run_seed_ga.py" \
  --seeds $SEEDS_DB \
  ${SEEDS_FILES:+--seed-files $SEEDS_FILES} \
  --seed-mac-cap "${SEED_MAC_CAP:-$MAC_CAP}" \
  --dataset "$DATASET" \
  --pop-size "$POP" --gens "$GENS" --epoch "$EP" \
  --elitism 5 --tournament 3 \
  --crossover block \
  --skip-ops \
  \
  `# --predictor --predict-keep 0.5 --predict-min-train 6 -- DISABLED, see header`  \
  --edge-filter --mac-cap "$MAC_CAP" \
  --diversity-gate \
  --cost-metric macs --cost-budget 0 \
  $EXTRA \
  --num-workers 12 \
  --checkpoint "$D/ga_${V}_checkpoint.pkl" \
  --cache      "$D/ga_${V}_cache.json" \
  --models-dir "$D/ga_${V}_models" \
  --results    "$D/ga_${V}_results.json" \
  --save-to-db

echo
echo "=== GA VARIANT=$V COMPLETE $(date -Is) ==="
echo "private DB: $GADB/ab.nn.db"
echo "Merge into the master when ready (single writer, dry-run first):"
echo "  python3 $D/merge_shard_dbs.py --main /a/mm/nn-dataset/db/ab.nn.db \\"
echo "      --shards $GADB/ab.nn.db --dry-run"
