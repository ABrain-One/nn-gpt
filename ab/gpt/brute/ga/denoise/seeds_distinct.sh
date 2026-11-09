# Single source of truth for the DISTINCT denoise GA seed pool (2026-07-27).
# Derived rigorously: all 29 edge-legal seeds embedded with arch_embed.embed_source,
# clustered at cosine 0.95 (the GA's own diversity-gate threshold), then the obvious
# clones dropped by hand -> 24 distinct families (7 DB + 17 new). Sourced by
# launch_distinct10.sh / launch_distinct_full.sh / populate_db.sh so the lists never
# drift between scripts.
#
# Edge-legal? every entry <= 9053 MMac (over-cap 2F/4F/7F/10F excluded upstream).
# Clones dropped (both from new_seeds): the norm cluster InstNorm/LayerNorm/ResBN/StatNorm
# (near-identical to GroupNorm, cos 0.94-0.998 -> keep GroupNorm as the representative)
# and SplitterNetL (cos 0.995 to SplitterNet -> keep SplitterNet + the distinct SplitterNetF).
D=/a/mm/nn-gpt/ab/gpt/brute/ga/denoise
NS="$D/new_seeds"

# --- 7 DB-resident seeds (resolved by --seeds; the 6 edge-legal F + Lite) ---
DB_SEEDS="DenoiseUnet1F DenoiseUnet3F DenoiseUnet5F DenoiseUnet6F DenoiseUnet8F DenoiseUnet9F LiteDenoiseNet"

# --- 17 distinct NEW families (file seeds; the 22 new_seeds minus 4 norm clones + SplitterNetL) ---
NEW_DISTINCT_FILES="\
$NS/DenoiseChanShuffle.py $NS/DenoiseDilatedFlat.py $NS/DenoiseELUCtx.py $NS/DenoiseGroupNorm.py \
$NS/DenoiseLPPool.py $NS/DenoiseLapPyr.py $NS/DenoiseLinearMix.py $NS/DenoiseMBConv.py \
$NS/DenoiseMFDNet.py $NS/DenoiseParallelMR.py $NS/DenoisePoolDeconv.py $NS/DenoiseReflectPad.py \
$NS/DenoiseSelfAttn.py $NS/DenoiseSplitterNet.py $NS/DenoiseSplitterNetF.py $NS/DenoiseSubPixel.py \
$NS/DenoiseThunder.py"

# --- BEST-10 subset for the first try: the 7 measured DB seeds (PSNR 33.2-36.5) plus the
#     three literature-proven efficient denoisers (MFDNet, Thunder, SplitterNet = the CVPR2024
#     target). These 3 are the strongest-pedigree NEW families; the rest await the populate_db
#     pass for measured PSNR before they can be ranked "best". ---
BEST10_DB="$DB_SEEDS"
BEST10_FILES="$NS/DenoiseMFDNet.py $NS/DenoiseThunder.py $NS/DenoiseSplitterNet.py"

# --- BEST-10 v2 (2026-07-30): re-derived on MEASURED PSNR, one model per family ---
# The list above was an informed guess made before the families had numbers -- it says so
# itself ("await the populate_db pass ... before they can be ranked best"). The canonical
# 50-epoch sweep has since measured all 18, and it inverts the guess:
#
#   in BEST10 v1 but among the WORST measured : DenoiseUnet3F 33.18, DenoiseUnet6F 33.76,
#                                               DenoiseUnet1F 34.37  (bottom 3 of 25)
#   excluded from v1 but the TOP measured     : DenoiseSubPixel 35.25, DilatedFlat 35.15,
#                                               LPPool 35.12, ELUCtx 35.11, MBConv 35.08
#
# v1 is also not "10 different families": 6 of its 10 are DenoiseUnet*F variants of one
# U-Net lineage. v2 keeps a single representative of that lineage (9F, the best at 35.43)
# and fills the rest with the top-scoring distinct families, all from the 24-family
# cosine-0.95 distinct set above, all edge-legal.
#
# dB shown is the mean of the last 5 of 50 epochs (NOT the peak: a max over 50 epochs
# rewards volatility -- SubPixel peaks 0.34 dB above where it settles).
#
#   DenoiseUnet9F 35.43   DenoiseSubPixel 35.25   DenoiseDilatedFlat 35.15
#   DenoiseLPPool 35.12   DenoiseELUCtx   35.11   DenoiseMBConv      35.08
#   DenoisePoolDeconv 35.04  DenoiseLinearMix 35.00  DenoiseSelfAttn 34.98
#   DenoiseThunder 34.98
#
# CAVEAT, recorded so nobody over-reads the ranking: these 10 span 0.45 dB, and the
# measured within-model epoch-to-epoch jitter is 0.14 dB. Ranks 2-10 are not separable;
# only "top group vs DenoiseUnet3F/6F" is. The point of v2 is to stop seeding the GA with
# the measured-worst models, not to claim SubPixel beats Thunder.
#
# All 10 carry DB rows (the populate_db pass), so --seeds resolves them; no --seed-files.
BEST10_V2_DB="DenoiseUnet9F DenoiseSubPixel DenoiseDilatedFlat DenoiseLPPool \
DenoiseELUCtx DenoiseMBConv DenoisePoolDeconv DenoiseLinearMix DenoiseSelfAttn \
DenoiseThunder"

# --- ALL 22 new families (for populate_db.sh: greenlit -> persist every legal family to DB) ---
ALL_NEW_FILES="$(ls "$NS"/*.py)"

# --- BEST10 v3 (2026-07-31): NORM-DIVERSE, measured, all mobile-clean -----------------
# v2 ranked purely on PSNR and produced a monoculture: 9 of its 10 seeds are norm-free and
# the 10th (DenoiseUnet9F, BatchNorm) dies on NaN and contributes nothing. Measured over the
# GA's 89 generated sources: 88 norm-free, 1 BatchNorm. Normalisation -- a real architectural
# axis -- is effectively frozen, because no mutation operator touches it either
# (BRUTE_WEIGHTS covers dimension/activation/layer_type/kernel_size/stride/scale/skip).
#
# v3 keeps the top norm-free performers and buys the axis back with two additions:
#   DenoiseUNet2  NORM-FREE, 35.368 last10 -- the BEST model measured, and it was invisible
#                 until 2026-07-31 because it had no rows in the working DB. Not a diversity
#                 concession: it outranks everything in v2.
#   DenoiseUNet   BatchNorm, same lineage -- gives a real BatchNorm lineage that trains
#                 cleanly, replacing DenoiseUnet9F (NaN 0-for-2, and pinned at 9052.6 of the
#                 9053 MMac cap so it cannot grow either).
#
# Both are ConvTranspose2d(output_padding=0) with no LayerNorm/GroupNorm/InstanceNorm, so
# both pass the HARD edge rule and neither trips --edge-strict. Confirm with
# edge_deploy.deployability() in-container before a production run; the check above was
# source-level.
#
# *** RETRACTION 2026-07-31, measured in-container by the NORM-v3 GA seed audit ***
# BOTH UNet entries above are ILLEGAL and were dropped by the GA at startup:
#     DROP over-cap seed DenoiseUNet2:  54618 MMac > 9053 cap
#     DROP over-cap seed DenoiseUNet:   55692 MMac > 9053 cap
# They are ~6x the edge budget. The paragraph above checked the HARD edge rule (transpose
# padding) and the norm modules and never checked MACs -- a source-level read cannot see
# them. So "DenoiseUNet2 is the best model we own" is true and irrelevant: at 54.6 GMac it
# was never competing in the same class as the 1.3-4.4 GMac families, and its 35.368 dB
# should NOT be quoted as a target for edge models. Rule that follows: no seed enters any
# list here until mac_audit.py has printed a number for it.
#
# What the GA actually ran with (9 seeds, and the norm balance survives the drop because
# DenoiseResBN carries BatchNorm):
#     norm-free 5  SubPixel LPPool ELUCtx Thunder SelfAttn
#     BatchNorm 1  DenoiseResBN
#     GroupNorm 1  DenoiseGroupNorm
#     LayerNorm 1  DenoiseLayerNorm
#     InstanceNorm 1  DenoiseInstNorm
# Bonus the drop exposed: ResBN/GroupNorm/LayerNorm/InstNorm all measure 3756.0 MMac and
# 0.73 Mp -- they are one backbone with only the norm swapped (the cos 0.94-0.998 "norm
# clone" cluster noted at the top of this file). For a norm experiment that is not
# redundancy, it is a controlled 4-way contrast.
#
# NOT included on purpose: DenoiseGroupNorm / DenoiseLayerNorm / DenoiseInstNorm and the
# plain DenoiseUnet9 (LayerNorm) / DenoiseUnet10 (GroupNorm). They are the only non-BN,
# non-norm-free models we own, but all three norms force a CPU fallback on the TFLite GPU
# delegate. Train them for the dataset if you want those data points; do NOT seed an
# edge-targeted search with them.
BEST10_V3_DB="DenoiseUNet2 DenoiseUNet DenoiseSubPixel DenoiseDilatedFlat DenoiseLPPool \
DenoiseELUCtx DenoiseMBConv DenoisePoolDeconv DenoiseSelfAttn DenoiseThunder"
