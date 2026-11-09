# Graph Report - .  (2026-08-05)

## Corpus Check
- Large corpus: 16382 files · ~226,144,524 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1137 nodes · 1594 edges · 134 communities (88 shown, 46 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.71)
- Token cost: 130,691 input · 0 output

## Community Hubs (Navigation)
- Architectural UID v2 (AST Dataflow)
- Architecture Embedding & Similarity
- Diversity Gate & Arm Diversity
- Functional Similarity (CKA)
- LLM Denoise Generator
- Preflight Candidate Validation
- Elite Archive & MAC Accounting
- Edge Cost & GA Seed Runner
- Skip Operators (AST Rewrites)
- GA Core Strategies & Template
- Arch-UID Backfill & Filtering
- GA Engine & Tournament Selection
- MAC Cap Audit & Fail-Open Hazard
- Block Crossover (AST Splice)
- Full-Image Eval & Noise Sweep
- Skip Inventory Analysis
- Corpus Clustering
- GA Design Notes & Norm Census
- Edge Deployability & Full Training
- LLM Run Notes & Prompt Defects
- Seed Mutation Scheduling
- Torch to TFLite Conversion
- Arch-UID v1 & Audit
- LLM Screening Run Records
- Predictor & Diversity Calibration Notes
- Mobile Denoiser Literature Baselines
- Screen Result Analysis
- Growth Loop Supervisor
- Scale Recalibration
- Run-3 SFT Dataset Builder
- Diversity Threshold Calibration
- Production Resume & Checkpointing
- ONNX to TFLite Path
- Prof Corpus Embedding
- Run Status Reporting
- GA Population Container
- LEMUR Stack & Repo Layout
- Dataset Manifest & Provenance
- GA Variant Launcher
- Predictor Tracking Figure
- Seed Runs Module
- Screen And Train Module
- Seed Runs Module
- Tflite Pool Convert Module
- Analyse Canonical Module
- Launch Beat1 Module
- Launch Norm V3 Module
- Llm Gen Job Module
- Norm Census Module
- Readme Module
- Bench Champions Module
- Benchmark Headroom Module
- Build Kto Dataset Module
- Launch A0 Module
- Launch B0 Module
- Launch C0 Module
- Launch D0 Module
- Launch Distinct10 Module
- Launch Distinct10 V2 Module
- Launch Distinct Full Module
- Launch E0 Module
- Launch Production Module
- Launch Production V2 Module
- Launch Run3 Module
- Launch Run4 Module
- Launch Run5 Module
- Launch Smoke Prod Module
- Llm Gen2 Module
- Llm Smoke Module
- R5 Report Module
- Repair Device Attr Module
- Rescue Test Module
- Run3 Evolve Module
- Run Seed Ga Module
- Track Autorefresh Module
- Accumulate Labels Module
- Export Mobile Job Module
- Launch Fulltrain Module
- Launch Wlval Module
- Llm Evolve Module
- Llm Evolve Par Module
- Mac Audit Module
- Mkjob Module
- Run Seed Ga Module
- Sweep Canonical Module
- Sweep Control 9F Module
- Sweep Recipe Module
- Sweep Unet Orig Module
- Sweep Valcrops Module
- Track Predictor Module
- Verify Epoch Stats Module
- Awgn Derisk Module
- Fulltrain P2 Supervisor Module
- Fulltrain Supervisor Module
- Generalize Sidd Module
- Import Canonical Stats Module
- Populate Db Module
- Rejection Digest Module
- Run3 Accept Module
- Run3 Db Import Module
- Run3 Sft Module
- Run4 Supervisor Module
- Run5 Supervisor Module
- Run Promote Chain Module
- Sidd Sweep Module
- Similarity Report Module
- Smoke Weights Module
- Tflite Export Job Module
- Train One Module
- Watch Gen Module
- Both Done Wait Module
- Build Prm Map Module
- Calib Job Module
- Diag Candidates Module
-   Init   Module
- Llm Div Job Module
- Llm Screen Shard Module
- Llm Stage2 Module
- Orchestrate Bd Module
- Preflight Job Module
- Production Supervisor Module
- Production V2 Supervisor Module
- Promote Chain Supervisor Module
- Readme Module
- Run2 Done Wait Module
- Run2 Marker Wait Module
- Run3 Supervisor Module
- Seeds Distinct Module
- Watch Final Module
- Watch Gen5 Module
- Watch Promote Module
- Watch Runs Module
- Wlval Supervisor Module
- Wlval Wait Module

## God Nodes (most connected - your core abstractions)
1. `DiversityGate` - 30 edges
2. `_Dataflow` - 19 edges
3. `embed()` - 18 edges
4. `both_source()` - 14 edges
5. `main()` - 14 edges
6. `run_seed_ga.py — denoise GA entry point` - 14 edges
7. `_Graph` - 13 edges
8. `_mac_hooks()` - 13 edges
9. `GeneticAlgorithmEngine` - 12 edges
10. `main()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Single-production-process rule` --semantically_similar_to--> `Single-writer SQLite-on-NFS rule`  [INFERRED] [semantically similar]
  RESUME_PRODUCTION.txt → SEED_RUNS.md
- `kMAC/pixel normalisation` --semantically_similar_to--> `TFLite latency caveat (desktop CPU, not phone ms)`  [INFERRED] [semantically similar]
  SEED_RUNS.md → README.md
- `elite_archive.py — diversity-gated elite pool with seed floor` --semantically_similar_to--> `Accuracy predictor pre-screen funnel (--predictor)`  [INFERRED] [semantically similar]
  SEED_RUNS.md → README.md
- `ep5 → ep50 Spearman +0.263` --semantically_similar_to--> `Proxy vs. final reporting rule`  [INFERRED] [semantically similar]
  SEED_RUNS.md → README.md
- `chain B/D status log — arm B auto-resume` --semantically_similar_to--> `production_supervisor.sh — crash-resilient wrapper`  [INFERRED] [semantically similar]
  chain_BD_status.txt → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GA search loop: seeds → operators → proxy eval → edge filter → rank** — readme_run_seed_ga, readme_ga_core, readme_seed_mutation, readme_block_crossover, readme_check_nn, readme_edge_filter, readme_accuracy_predictor, readme_fitness_eff [EXTRACTED 1.00]
- **LLM funnel: generate → CPU preflight → 1-epoch screen → 5-epoch train → elite archive** — seed_runs_llm_denoise, seed_runs_preflight, seed_runs_screen_and_train, seed_runs_analyse_screen, seed_runs_elite_archive, seed_runs_llm_evolve [EXTRACTED 1.00]
- **MAC budget governance: measure, never inherit; unmeasurable is not legal** — seed_runs_mac_cap_9053, seed_runs_mac_audit, seed_runs_fail_open_hazard, seed_runs_record_measured_macs_rule, seed_runs_denoiseunet2_retraction, seed_runs_llm_mac_gap, seed_runs_mac_cap_literature_check [EXTRACTED 1.00]

## Communities (134 total, 46 thin omitted)

### Community 0 - "Architectural UID v2 (AST Dataflow)"
Cohesion: 0.06
Nodes (48): adjacency(), arch_similarity(), arch_uid_and_vec(), arch_vec(), authored_region(), _bucket(), _const_eval(), _Dataflow (+40 more)

### Community 1 - "Architecture Embedding & Similarity"
Cohesion: 0.07
Nodes (48): _adjacency(), _behav_node_features(), _behaviour(), both_source(), _bucket(), _cleanup(), _combine(), _descriptor() (+40 more)

### Community 2 - "Diversity Gate & Arm Diversity"
Cohesion: 0.05
Nodes (38): arm_of(), main(), Which generator actually supplies the architectural variety? A cumulative…, DiversityGate, Exact hash from the AST of EVERY method; graded embedding from the fx trace.…, Return (accept: bool, fp: (hash, embed)). accept=True when the candidate is a…, Add an accepted model's fingerprint to the reference set (greedy) + persist., -> (hash, embed). (None, None) when the model cannot be traced. THE except IS… (+30 more)

### Community 3 - "Functional Similarity (CKA)"
Cohesion: 0.09
Nodes (36): arm_of(), main(), Are our models FUNCTIONALLY different, or only structurally different? WHAT WE…, _disable_db_prm_lookup(), get_noisy_batch(), gram_centered(), linear_cka(), log() (+28 more)

### Community 4 - "LLM Denoise Generator"
Cohesion: 0.10
Nodes (36): _by_arm(), canonical_dir(), check_contract(), compose(), db_corpus(), db_path(), _edge_checker(), _extract_block() (+28 more)

### Community 5 - "Preflight Candidate Validation"
Cohesion: 0.08
Nodes (28): check_isolated(), check_one(), _class_defs(), main(), CPU pre-flight: reject candidates that cannot build or forward, BEFORE spending…, {name: source_segment} for every top-level class. Only classes -- splicing a…, Class definitions from the REFERENCE code in this candidate's own prompt. The…, -> (src, [spliced]). Splice in helper classes the candidate uses but never… (+20 more)

### Community 6 - "Elite Archive & MAC Accounting"
Cohesion: 0.08
Nodes (31): _mac_hooks(), Register MAC-counting hooks; return (handles, macs_accumulator)., collect(), _common_depth(), curve_of(), floor_seeds(), macs_mm(), main() (+23 more)

### Community 7 - "Edge Cost & GA Seed Runner"
Cohesion: 0.10
Nodes (28): _cleanup(), effective_db(), measure_cost(), Edge-cost measurement + PSNR/cost objective for the denoise GA. The thesis…, Instantiate a LEMUR denoise Net from source; return its edge cost. Returns…, Combine PSNR (dB) and edge cost into a single fitness score (dB units).…, macs_ok(), True if macs (raw MACs) is within the device cap (in MMac). cap<=0 disables. (+20 more)

### Community 8 - "Skip Operators (AST Rewrites)"
Cohesion: 0.13
Nodes (29): add_block_residual(), add_global_residual(), add_lateral_skip(), build_ok(), _builds(), _builds_and_trains(), _classes(), _first_arg() (+21 more)

### Community 9 - "GA Core Strategies & Template"
Cohesion: 0.12
Nodes (17): ABC, Skip-configurable denoise U-Net template. ``render_model(chromosome)`` returns…, Render a standalone Net source string from a gene dict., render_model(), CrossoverStrategy, MutationStrategy, RandomResettingMutation, Generic GA core — copied verbatim (behaviour-preserving) from ga/modular/.… (+9 more)

### Community 10 - "Arch-UID Backfill & Filtering"
Cohesion: 0.11
Nodes (21): main(), Add an ARCHITECTURAL id to the nn table, alongside the existing lexical one.…, arch_uid(), Exact architectural certificate for a model given as source text., main(), of_source(), Fill nn.arch_uid for rows that do not have one yet. Runs after every merge. WHY…, -> (uid, error), preserving the old interface's explicit-failure contract. (+13 more)

### Community 11 - "GA Engine & Tournament Selection"
Cohesion: 0.16
Nodes (7): GeneticAlgorithmEngine, Individual, pre_generation_hook(individuals, gen) is called once per generation BEFORE the…, DiverseTournamentSelection, Mutate a chromosome's source with one random brute AST op. Retries until it…, Tournament for the FIRST parent (unchanged fitness pressure); the SECOND parent…, SeedMutation

### Community 12 - "MAC Cap Audit & Fail-Open Hazard"
Cohesion: 0.13
Nodes (18): edge_cost.py — MAC / cost measurement, edge_deploy.py — deployability filter, Edge deployability filter (--edge-filter, --mac-cap), to_tflite.py — TFLite conversion and latency measurement, DenoiseUNet2 edge-legality retraction, DenoiseUnet4F — over-cap seed missed by the denylist, DenoiseUnet9F seed (9053 MMac, 35.43 dB), Fail-open MAC gate hazard (+10 more)

### Community 13 - "Block Crossover (AST Splice)"
Cohesion: 0.16
Nodes (12): _closure(), crossover(), _init_args(), _module_classes(), Component-block-swap crossover for the denoise seed GA (arm C recombination).…, Child = A's skeleton with A's conv block swapped for B's (renamed to A's block…, Name of the class whose __init__ is exactly (self, channels) and that contains…, Top-level identifiers referenced anywhere in node (Names + Call/Base ids). (+4 more)

### Community 14 - "Full-Image Eval & Noise Sweep"
Cohesion: 0.16
Nodes (15): main(), psnr_from_mses(), Do our models actually DENOISE, and do they do it on a whole image? TWO…, Mean squared error on the ITU-R BT.601 LUMINANCE channel, matching…, dB of the MEAN MSE -- the metric averages MSE over samples and converts ONCE.…, Run `net` over a large image in overlapping tiles, keeping only tile interiors., tiled_forward(), y_mse() (+7 more)

### Community 15 - "Skip Inventory Analysis"
Cohesion: 0.19
Nodes (16): _added_output(), analyse(), _arch_forwards(), _has_block_residual(), _has_dense(), _has_global_residual(), _has_lateral(), _is_arch_class() (+8 more)

### Community 16 - "Corpus Clustering"
Cohesion: 0.17
Nodes (13): main(), Merge embed_corpus.py shards and report distinct-architecture counts. CPU, no…, gate_cluster(), main(), Three-way comparison on the LEMUR zoo: fx (ours), AST (his), HYBRID (the live…, Replay DiversityGate.check/add over `order`. -> (n_accepted, n_exact,…, cluster(), load_dir() (+5 more)

### Community 17 - "GA Design Notes & Norm Census"
Cohesion: 0.14
Nodes (15): block_crossover.py — block-level source recombination, Fitness eff = PSNR − 0.3·ln(MACs / budget), ga_core.py — generic GA engine, MAI search dataset (ab/nn/loader/denoise.py), run_seed_ga.py — denoise GA entry point, seed_mutation.py — mutation operators, skip_operators.py — skip-connection operators, Static audits of generated code need AST, not regex (+7 more)

### Community 18 - "Edge Deployability & Full Training"
Cohesion: 0.22
Nodes (11): deployability(), _instantiate(), Fast STATIC edge-deployability check — gate models at GENERATION time. Thesis…, exec the source and build Net on CPU. Returns the nn.Module or None., Scan the module tree for GPU-delegate-hostile ops. Returns {deployable: bool,…, main(), Gate 1 — full-train the top-K per ablation arm at 50 epochs (the PROMOTION…, Return an ordered list of (checksum, rec) to full-train for this arm: top-K by… (+3 more)

### Community 19 - "LLM Run Notes & Prompt Defects"
Cohesion: 0.15
Nodes (13): DenoiseUnet1F..10F / LiteDenoiseNet seed architectures, Hardcoded /home/s493053/thesis path portability issue, promote.py — grow the seed pool from winners, LLM run denoise-llmgen-1 (2026-07-30), GA vs LLM throughput (models into DB per GPU-hour), llm_denoise.py — LLM architecture generator, LLM EVOLVE — generation N+1 conditioned on N's survivors, The LLM has no generations, by construction (+5 more)

### Community 20 - "Seed Mutation Scheduling"
Cohesion: 0.21
Nodes (11): _checksum(), _cleanup(), mutate_source(), Mutate a real LEMUR model *source string* with the brute AST operators.…, Return a mutated source string (one random op), or None on failure. If…, Translate a plan into scheduled CodeMutator edits (copied dispatch)., Instantiate the emitted source and check output shape == input shape., _schedule() (+3 more)

### Community 21 - "Torch to TFLite Conversion"
Cohesion: 0.26
Nodes (12): _cleanup(), convert_one(), _disable_db_prm_lookup(), load_torch(), load_weights(), log(), main(), Convert denoise models to TFLite (LiteRT) and check their edge inference. The… (+4 more)

### Community 22 - "Arch-UID v1 & Audit"
Cohesion: 0.23
Nodes (10): arm_of(), main(), How many of our models are ISOMORPHIC DUPLICATES? Measure it before changing…, build(), of_net(), of_source(), THE canonical architectural id. One construction, used by every producer and…, Instantiate a candidate the canonical way. Raises on failure. (+2 more)

### Community 23 - "LLM Screening Run Records"
Cohesion: 0.17
Nodes (12): Single-production-process rule, analyse_screen.py — pool and analyse screen results, Arm baked into the DB model name, --check-contract prompt-identity verification, Cold vs seeded prompting arms, dataset_manifest.py — provenance labelling and git delta, git add -A deletion hazard (252,227 deleted files), LLM run denoise-llmgen2 (516 parseable of 550) (+4 more)

### Community 24 - "Predictor & Diversity Calibration Notes"
Cohesion: 0.18
Nodes (11): accumulate_labels.py — predictor label store refresh, Accuracy predictor pre-screen funnel (--predictor), Missing behavioural (CKA) diversity anchor, calibrate_diversity.py — re-derive the cosine threshold, Quote single-linkage cluster count, not greedy accepts, curve_from JSON-shape bug (silent nan last5), Degenerate positive anchor in diversity calibration, DiversityGate — WL graph-hash novelty gate (+3 more)

### Community 25 - "Mobile Denoiser Literature Baselines"
Cohesion: 0.22
Nodes (11): TFLite latency caveat (desktop CPU, not phone ms), DenoiseThunder — most defensible edge model owned, kMAC/pixel normalisation, The 9053 MMac cap does not survive a literature check, MAI 2026 NPU denoising (arXiv 2605.03680), MFDNet / MFDNet-S / MFDNet-L (arXiv 2211.04687), Mobile AI 2021 denoising challenge (arXiv 2105.08629), NAFNet (3.81 G @720p) (+3 more)

### Community 26 - "Screen Result Analysis"
Cohesion: 0.36
Nodes (9): arm_of(), block(), fail_reasons(), load(), main(), Pool the sharded 1-epoch screen results and report what the 516 LLM candidates…, ids whose candidate.py we patched -- meta.json carries the marker., repaired_ids() (+1 more)

### Community 27 - "Growth Loop Supervisor"
Cohesion: 0.36
Nodes (7): BRUTE_WEIGHTS, launch_run(), log(), promote_run(), PYTHONPATH, growth_loop.sh script, wait_for_run()

### Community 28 - "Scale Recalibration"
Cohesion: 0.33
Nodes (9): describe(), is_crossover(), load_lineage(), main(), pct(), Does the 0.95 threshold still hold now that we have 5,000 models instead of a…, DenoiseThunder+m+s+m' -> 'DenoiseThunder'. Crossovers keep both parents., {hash: {name, parent, src}} from every GA run's cache + models dir. (+1 more)

### Community 29 - "Run-3 SFT Dataset Builder"
Cohesion: 0.29
Nodes (9): corpus_targets(), _dev_split(), main(), Run-3 SFT dataset: score-conditioned generation examples from the measured…, Deterministic membership by name hash -- stable across cycles, so an appended…, Best-scored row per model at/above the bar; one representative per arch class., One SFT example per target, in the REPO's improvement format…, render() (+1 more)

### Community 30 - "Diversity Threshold Calibration"
Cohesion: 0.33
Nodes (8): greedy_count(), main(), pct(), Calibrate a diversity threshold FOR LLM OUTPUT, instead of inheriting the GA's.…, [(name, hash, embed)] for [(name, source)]., Order-independent: union-find over all pairs with cosine >= thr., single_linkage_clusters(), trace_all()

### Community 31 - "Production Resume & Checkpointing"
Cohesion: 0.28
Nodes (9): chain B/D status log — arm B auto-resume, launch_B0.sh — ablation arm B launcher, Per-generation checkpointing and evaluation cache, launch_production.sh — canonical RUN 1 configuration, production_supervisor.sh — crash-resilient wrapper, Production run resume procedure, seed_ga_P1 run namespace (checkpoint + cache), track_autorefresh.sh — predictor tracking monitor (+1 more)

### Community 32 - "ONNX to TFLite Path"
Cohesion: 0.33
Nodes (8): convert_one(), _from_saved_model(), main(), Stage 2 of the mobile export: ONNX -> TFLite, via onnx2tf. NEVER IMPORTS TORCH.…, Run the flatbuffer once so a file that exists but cannot load is not reported…, Re-convert the SavedModel with the stock TFLiteConverter. Needed for two…, -> dict describing what was produced. Raises nothing; reports instead., verify()

### Community 33 - "Prof Corpus Embedding"
Cohesion: 0.25
Nodes (4): main(), Embed one CONTIGUOUS SHARD of a LEMUR task under the AST metric. Pairs with…, Per-model wall-clock ceiling. WALL CLOCK, NOT RLIMIT_CPU, and the difference is…, _Timeout

### Community 34 - "Run Status Reporting"
Cohesion: 0.39
Nodes (8): corpus(), ga_status(), llm_status(), main(), _pod_phase(), One status line per run for every generator we have in flight. Login-node safe.…, Phase of the newest pod for a job, or None when kubectl is unavailable., _tail()

### Community 36 - "LEMUR Stack & Repo Layout"
Cohesion: 0.29
Nodes (8): Denoise NAS — GA architecture search for mobile image denoising, ABrain-One LEMUR stack, nn-dataset (datasets, Train.py, metrics, seeds, ab.nn.db), nn-gpt ab/gpt/brute AST mutation engine, Artifact-level (not bit-exact) reproducibility, Thesis: Towards Automated Image Denoising Model Generation for Resource-Constrained Edge Systems, Two-sibling-repo layout under THESIS_ROOT, out_shape=(1000,) classification-default trap

### Community 37 - "Dataset Manifest & Provenance"
Cohesion: 0.43
Nodes (6): git(), llm_arm_map(), main(), provenance(), Label every denoise model with full provenance, and report what git has NOT…, LLM_Denoise DB name -> 'cold'/'seeded', joined via the eval results.

### Community 38 - "GA Variant Launcher"
Cohesion: 0.29
Nodes (6): BRUTE_DEBUG, BRUTE_NORM_SAFE, BRUTE_WEIGHTS, LEMUR_CKPT_DIR, PYTHONPATH, launch_ga_variant.sh script

### Community 39 - "Predictor Tracking Figure"
Cohesion: 0.62
Nodes (7): Predictor vs Actual ep5 PSNR Tracking Scatter, GA / Brute-Force Denoise Architecture Search Population (n=300), y=x Perfect-Prediction Reference Line, Denoise Performance Predictor (ep5 PSNR), Predictor Fails to Track Actual Performance (Anti-correlated, Systematic Under-prediction), Epoch-5 PSNR (dB) Denoising Metric, Spearman Rank Correlation rho = -0.18

### Community 40 - "Seed Runs Module"
Cohesion: 0.33
Nodes (7): check_nn — LEMUR training/eval entry point, PSNR metric with PSNR_SCALE = 48, The screen gate is a label, not a filter, Pod-local LEMUR root (shared scratch dir fix), screen_and_train.py — 1-epoch screen + staged training, Single-writer SQLite-on-NFS rule, Strided (i % num_shards) candidate sharding

### Community 41 - "Screen And Train Module"
Cohesion: 0.38
Nodes (6): arm_of(), curve_from(), main(), Two-stage evaluation of LLM candidates: 1-epoch screen, then full train into…, [(epoch, dB)] from the per-epoch JSONs check_nn writes., cold' / 'seeded' from the candidate id; '' when it cannot be told.

### Community 42 - "Seed Runs Module"
Cohesion: 0.29
Nodes (7): BEST10_DB (superseded seed list), BEST10_V2_DB seed list (seeds_distinct.sh), Record the dB definition (last5 vs best), DenoiseSubPixel — best hand-written family (35.25 dB), Collapse architectural families before taking a top-N, One-shot LLM approach stopped (all models 1.3–2.2 dB below hand-written), Seed-set ledger — which seeds fed which run

### Community 43 - "Tflite Pool Convert Module"
Cohesion: 0.43
Nodes (6): convert_one(), dump_codes(), main(), Pool-wide TFLite FP32 conversion sweep for the denoise corpus. Converts every…, Worker mode: exec + instantiate + forward-check + convert. Prints one JSON line., run_shard()

### Community 44 - "Analyse Canonical Module"
Cohesion: 0.47
Nodes (5): curve(), main(), Post-sweep analysis of the canonical 50-epoch run. Answers the two questions…, {epoch: dB} for one model from one DB file., spearman()

### Community 45 - "Launch Beat1 Module"
Cohesion: 0.33
Nodes (5): BRUTE_DEBUG, BRUTE_WEIGHTS, CUDA_VISIBLE_DEVICES, PYTHONPATH, launch_beat1.sh script

### Community 46 - "Launch Norm V3 Module"
Cohesion: 0.33
Nodes (5): BRUTE_DEBUG, BRUTE_NORM_SAFE, BRUTE_WEIGHTS, PYTHONPATH, launch_norm_v3.sh script

### Community 47 - "Llm Gen Job Module"
Cohesion: 0.40
Nodes (5): HF_HOME, HF_HUB_DISABLE_TELEMETRY, log(), PYTHONPATH, llm_gen_job.sh script

### Community 48 - "Norm Census Module"
Cohesion: 0.47
Nodes (5): census(), main(), norms_of(), Count which normalisation layers a corpus of models actually INSTANTIATES. WHY…, Set of norm families CONSTRUCTED in this file, or None if unparseable.

### Community 49 - "Readme Module"
Cohesion: 0.40
Nodes (6): fulltrain_topk.py — 50-epoch full training of top-K, 5-epoch proxy training screen, Proxy vs. final reporting rule, The DB records the final epoch, not the peak, ep5 → ep50 Spearman +0.263, Stage-2 epoch budget corrected 50 → 5

### Community 50 - "Bench Champions Module"
Cohesion: 0.50
Nodes (4): cost(), main(), Standard-benchmark head-to-head for the top MAI/Lemur champions (professor…, params (M) and conv/linear MACs (MMac) for a 3x256x256 input, static — no…

### Community 51 - "Benchmark Headroom Module"
Cohesion: 0.50
Nodes (4): main(), pick(), Does a noisier benchmark separate architectures better? Same models, both…, Stratified by MAI score, plus every hand-written model. STRATIFIED, NOT RANDOM.…

### Community 52 - "Build Kto Dataset Module"
Cohesion: 0.50
Nodes (4): main(), Turn generation episodes into the KTO JSONL that TuneNNGenKTO consumes, plus a…, Drop trailing supporting REFERENCE blocks until the rendered prompt fits. The…, trim_prompt()

### Community 53 - "Launch A0 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_A0.sh script

### Community 54 - "Launch B0 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_B0.sh script

### Community 55 - "Launch C0 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_C0.sh script

### Community 56 - "Launch D0 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_D0.sh script

### Community 57 - "Launch Distinct10 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_distinct10.sh script

### Community 58 - "Launch Distinct10 V2 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_distinct10_v2.sh script

### Community 59 - "Launch Distinct Full Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_distinct_full.sh script

### Community 60 - "Launch E0 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_E0.sh script

### Community 61 - "Launch Production Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_production.sh script

### Community 62 - "Launch Production V2 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_production_v2.sh script

### Community 63 - "Launch Run3 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_run3.sh script

### Community 64 - "Launch Run4 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_run4.sh script

### Community 65 - "Launch Run5 Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_run5.sh script

### Community 66 - "Launch Smoke Prod Module"
Cohesion: 0.40
Nodes (4): BRUTE_DEBUG, BRUTE_WEIGHTS, PYTHONPATH, launch_smoke_prod.sh script

### Community 67 - "Llm Gen2 Module"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_DISABLE_TELEMETRY, PYTHONPATH, llm_gen2.sh script

### Community 68 - "Llm Smoke Module"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_DISABLE_TELEMETRY, PYTHONPATH, llm_smoke.sh script

### Community 70 - "Repair Device Attr Module"
Cohesion: 0.50
Nodes (4): main(), Repair the one-line harness-contract defect that killed 81% of the cold arm.…, Return (new_src, status)., repair()

### Community 71 - "Rescue Test Module"
Cohesion: 0.50
Nodes (4): instrumented_guard(), main(), NaN-guard rescue test (yield >90% proof). Plain DenoiseUnet9F NaN-diverged to…, Same guard as production but prints when it fires, so we can count rescues.

### Community 72 - "Run3 Evolve Module"
Cohesion: 0.90
Nodes (4): run_h100_job(), run_job(), run3_evolve.sh script, wait_job()

### Community 74 - "Track Autorefresh Module"
Cohesion: 0.50
Nodes (3): PYTHONPATH, run_tracker(), track_autorefresh.sh script

### Community 75 - "Accumulate Labels Module"
Cohesion: 0.67
Nodes (3): _load(), main(), Accumulate a GROWING label store for the v2 accuracy predictor. The v1…

### Community 76 - "Export Mobile Job Module"
Cohesion: 0.50
Nodes (3): HOME, export_mobile_job.sh script, XDG_CACHE_HOME

### Community 77 - "Launch Fulltrain Module"
Cohesion: 0.50
Nodes (3): BRUTE_DEBUG, PYTHONPATH, launch_fulltrain.sh script

### Community 78 - "Launch Wlval Module"
Cohesion: 0.50
Nodes (3): BRUTE_DEBUG, PYTHONPATH, launch_wlval.sh script

### Community 79 - "Llm Evolve Module"
Cohesion: 0.50
Nodes (3): HF_HOME, HF_HUB_DISABLE_TELEMETRY, llm_evolve.sh script

### Community 80 - "Llm Evolve Par Module"
Cohesion: 0.83
Nodes (3): run_job(), llm_evolve_par.sh script, wait_job()

### Community 81 - "Mac Audit Module"
Cohesion: 0.67
Nodes (3): main(), nn_dir(), Measure MACs for the denoise seed corpus against the 9053 MMac edge budget. WHY…

### Community 82 - "Mkjob Module"
Cohesion: 0.67
Nodes (3): build(), main(), Emit a K8s Indexed Job manifest for one stage of the parallel evolve loop. The…

### Community 83 - "Run Seed Ga Module"
Cohesion: 0.50
Nodes (4): epoch_stats(), Aggregate {epoch: accuracy} into the reported statistics block., Full per-epoch training statistics for an evaluated model. WHY THIS IS NOT JUST…, _summarise_curve()

### Community 84 - "Sweep Canonical Module"
Cohesion: 0.67
Nodes (3): log(), PYTHONPATH, sweep_canonical.sh script

### Community 85 - "Sweep Control 9F Module"
Cohesion: 0.67
Nodes (3): log(), PYTHONPATH, sweep_control_9f.sh script

### Community 86 - "Sweep Recipe Module"
Cohesion: 0.67
Nodes (3): log(), PYTHONPATH, sweep_recipe.sh script

### Community 87 - "Sweep Unet Orig Module"
Cohesion: 0.67
Nodes (3): log(), PYTHONPATH, sweep_unet_orig.sh script

### Community 88 - "Sweep Valcrops Module"
Cohesion: 0.67
Nodes (3): log(), PYTHONPATH, sweep_valcrops.sh script

### Community 89 - "Track Predictor Module"
Cohesion: 0.67
Nodes (3): main(), Track the accuracy predictor's LIVE performance on the production run. The…, _spearman()

### Community 90 - "Verify Epoch Stats Module"
Cohesion: 0.67
Nodes (3): log(), PYTHONPATH, verify_epoch_stats.sh script

## Ambiguous Edges - Review These
- `Epoch-5 PSNR (dB) Denoising Metric` → `GA / Brute-Force Denoise Architecture Search Population (n=300)`  [AMBIGUOUS]
  predictor_tracking_scatter.png · relation: conceptually_related_to

## Knowledge Gaps
- **163 isolated node(s):** `both_done_wait.sh script`, `calib_job.sh script`, `export_mobile_job.sh script`, `HOME`, `XDG_CACHE_HOME` (+158 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Epoch-5 PSNR (dB) Denoising Metric` and `GA / Brute-Force Denoise Architecture Search Population (n=300)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `DiversityGate` connect `Diversity Gate & Arm Diversity` to `Functional Similarity (CKA)`, `Preflight Candidate Validation`, `Elite Archive & MAC Accounting`, `Edge Cost & GA Seed Runner`, `Run Seed Ga Module`, `Arch-UID Backfill & Filtering`, `GA Engine & Tournament Selection`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `_disable_db_prm_lookup()` connect `Functional Similarity (CKA)` to `Arch-UID Backfill & Filtering`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `both_source()` connect `Architecture Embedding & Similarity` to `Functional Similarity (CKA)`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `DiversityGate` (e.g. with `_Timeout` and `DiverseTournamentSelection`) actually correct?**
  _`DiversityGate` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `both_done_wait.sh script`, `calib_job.sh script`, `export_mobile_job.sh script` to the rest of the system?**
  _163 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Architectural UID v2 (AST Dataflow)` be split into smaller, more focused modules?**
  _Cohesion score 0.06351236146632566 - nodes in this community are weakly interconnected._