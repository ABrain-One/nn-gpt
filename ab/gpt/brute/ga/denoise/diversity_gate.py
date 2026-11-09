"""WL-GRAPH-HASH DIVERSITY GATE for the denoise GA (Run 2+ anti-convergence fix).

Run 1 collapsed onto the LiteDenoiseNet basin: 291 of 300 usable models were Lite
mutants. A monoculture NAS defeats the thesis (denoising = a SEARCH SPACE of skip+conv
combos, not one fine-tuned family). This gate applies explicit counter-pressure at
generation time: reject a freshly GENERATED candidate whose ARCHITECTURE already exists
in the accepted set, BEFORE the (expensive) training.

Metric = exact WL graph hash from `arch_embed` (the unified VOCABULARY-FREE algorithm:
torch.fx trace; node identity = op-type + ALL generically-introspected numeric hyperparams,
no hand-authored op list). Equal hash => same architecture => clone. Exact, FREE (no forward
pass, no GPU, no weights), name-agnostic, seed-independent, topology-aware (skip wiring).

Chosen over residual-CKA after direct measurement: CKA on random-init models is weight-
noise-dominated (same architecture reloaded scored 0.69, distinct archs 0.92; could not
separate 1F from 9F). Functional CKA and empirical-NTK were both tested seed-marginalized
and do NOT transfer to denoising (reconstruction U-Nets share an encoder-decoder-skip
skeleton, so features AND gradient kernels saturate). See scratchpad/{wl_hash_test,
seedavg_cka,proxy_cka,ntk_align}.py and DRAFT_ntk_findings_dmytro.md.

Seeds (chromosome parent is None) are NEVER rejected — they are the fixed baselines — but
their graph IS added to the reference set, so a gen-1 child that merely re-enacts a seed's
graph is gated. Greedy: an accepted child joins the reference set so later children must
differ too. Resumable: accepted hashes are persisted so a restart re-seeds the ref set.

NOTE: exact-clone semantics. WL-hash rejects models with an IDENTICAL graph (renamed or
width-rescaled duplicates); two graphs differing by even one node are BOTH kept. There is
no reliable graded near-duplicate signal for denoising (functional metrics don't transfer),
so "distinct" here means "distinct computation graph".
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


class DiversityGate:
    # signature kept backward-compatible with run_seed_ga.py / the launchers; the CKA-era
    # args (cka_max, n_patches, batch_seed, device) are accepted but unused by the WL gate.
    def __init__(self, cka_max=None, n_patches=None, batch_seed=None, store_path=None,
                 log=print, device=None, graded=False, cos_max=0.95, ast_exact=False):
        # TWO operating points from arch_embed:
        #   EXACT hash  -> always on: reject literal graph-clones.
        #   GRADED cosine (opt-in, `graded`): also reject near-duplicates with continuous-WL
        #   embedding cosine >= cos_max. Threshold 0.95 CALIBRATED (metric_ablation): trivial
        #   variation (width + same-family mutants) sits >=0.985, distinct archs <=0.921. NOTE:
        #   for a mutation-only generator the graded gate is a cross-FAMILY forcer (mutants form
        #   a tight >=0.985 cluster) — enable only when the generator can leave the family.
        self.log = log
        self.store_path = store_path
        self.graded = graded
        self.cos_max = cos_max
        # OFF BY DEFAULT, and it must stay that way while runs are live: the persisted store
        # holds exact hashes in whichever space produced them, and mixing spaces silently
        # empties the reference set instead of failing. Enable only alongside a store built
        # from scratch with the same setting.
        self.ast_exact = ast_exact
        self.accepted = set()                    # set of exact WL graph hashes
        self.embeds = []                         # continuous-WL vectors (graded only)
        # WHICH model each embed belongs to. add() has always accepted a `key` and thrown it
        # away, so a rejection could only ever say "too similar to something" -- the gate knew
        # the score and the neighbour and reported neither. That makes the gate a silent
        # filter: the LLM has 32 candidates a generation refused with no way to learn what it
        # collided with. Keys are kept parallel to embeds so a rejection can name its match.
        self.embed_keys = []
        # Set by every check(): {"key", "cos", "kind"} for the closest accepted model, or None.
        self.last_match = None
        self.n_reject = 0
        self.n_accept = 0

        from functional_similarity import _disable_db_prm_lookup
        _disable_db_prm_lookup()                 # skip the ~2-min per-model DB prm scan

        if store_path and os.path.exists(store_path):
            try:
                import torch
                # store holds plain python/numpy objects; newer torch defaults
                # weights_only=True which rejects them -> resume would silently
                # start with an EMPTY reference set (near-dup re-admission).
                loaded = torch.load(store_path, weights_only=False)
                if isinstance(loaded, dict):     # new format {hashes, embeds}
                    self.accepted = set(loaded.get("hashes", []))
                    self.embeds = list(loaded.get("embeds", []))
                    # Stores written before keys were tracked have none. Pad rather than
                    # refuse: an older store is still a valid reference set, it just cannot
                    # name its matches until new acceptances refill the list.
                    self.embed_keys = list(loaded.get("embed_keys", []))
                    if len(self.embed_keys) < len(self.embeds):
                        self.embed_keys += [None] * (len(self.embeds) - len(self.embed_keys))
                else:                            # old format: bare set of hashes
                    self.accepted = set(loaded)
                log(f"  [diversity] resumed reference set: {len(self.accepted)} accepted graphs"
                    + (f", {len(self.embeds)} embeds" if self.graded else ""))
            except Exception as e:
                log(f"  [diversity] could not load {store_path} ({e}); starting empty")

    # -- fingerprint via arch_embed (single source of truth for both operating points) --
    def _fingerprint(self, code):
        """-> (hash, embed). (None, None) when the model cannot be traced.

        THE except IS LOAD-BEARING. check() documents "accept=True when the candidate is a
        NOVEL architecture (or untraceable)" and tests `if h is None`, but arch_embed raises
        on failure rather than returning None -- so that branch was unreachable and the
        exception escaped through run_seed_ga.py:565, which does not catch it. One model with
        data-dependent control flow (`if` on a traced value inside _pad) therefore killed an
        entire GA run: the `struct` variant died 40 s after its restart with
        "TraceError: symbolically traced variables cannot be used as inputs to control flow"
        and sat dead for 11 hours.

        Untraceable means "no architectural opinion", never "reject" and never "crash": the
        candidate is admitted and training judges it, which is exactly what the docstring
        already promised.
        """
        if self.ast_exact:
            return self._fingerprint_hybrid(code)
        import arch_embed
        try:
            if self.graded:
                return arch_embed.both_source(code)  # (hash, embed) — one trace
            return arch_embed.exact_source(code), None
        except Exception as e:
            self.log(f"  [diversity] untraceable candidate, admitting it "
                     f"({type(e).__name__}); training will judge")
            return None, None

    def _fingerprint_hybrid(self, code):
        """Exact hash from the AST of EVERY method; graded embedding from the fx trace.

        Each half comes from whichever backend measured better at its own job.

        EXACT -> AST (the professor's ast_model_graph, plus module-level class collection).
        torch.fx traces `forward` and nothing else, so optimizer, loss and the update loop
        never reach the identity. On this corpus that merges 38 groups whose training code
        differs -- including DenoiseUnet5 vs DenoiseUnet5F, which differ by MSE vs Charbonnier
        loss and clip 0.1 vs 0.5, and score 37.44 vs 34.76 dB. The AST hash also covers 100%
        of models against our 97.5%, because it never builds or runs anything.

        GRADED -> our continuous-WL embedding. Measured head-to-head on 4,536 lineage-labelled
        pairs, ours separates known-same from known-different at Youden J 0.941 against 0.849.
        The gap is concentrated where it matters most: crossover hybrids score a median 0.975
        under the AST embedding, indistinguishable from same-family mutants at 0.994, so it
        would suppress the one operator that escapes a family. Ours puts them at 0.900 against
        0.984.

        DEGRADED MODE IS THE POINT OF SPLITTING THEM. When the fx trace fails -- 2.5% here,
        24.6% on the LEMUR classification zoo -- the old code returned (None, None) and the
        candidate was admitted with no check at all. Now the AST hash still lands, so an exact
        clone is still caught; only the near-duplicate opinion is missing.
        """
        h = e = None
        try:
            # SWITCHED (2026-08-03) to arch_uid2: the standardised source-only identity
            # (v2 -- per-class attribute scoping, slice ports, chained receivers). Nothing
            # is imported or executed, so load_class's tempdir dance is gone, and the same
            # pure function now computes the identity here, in preflight, and in the DB
            # backfill. HASH-SPACE CAVEAT: a store written by the old prof-uid hash will
            # not match these values -- a resumed run with an old store silently behaves
            # as if the reference set were empty. Start fresh stores after this date.
            from arch_uid2 import arch_uid as _uid2
            h = _uid2(code)
        except Exception as ex:
            self.log(f"  [diversity] AST hash failed ({type(ex).__name__}); "
                     f"falling back to the fx hash for this candidate")
        if self.graded:
            try:
                import arch_embed
                fh, e = arch_embed.both_source(code)
                if h is None:
                    h = fh
            except Exception as ex:
                self.log(f"  [diversity] untraceable for the graded check "
                         f"({type(ex).__name__}); exact check still applies")
        elif h is None:
            try:
                import arch_embed
                h = arch_embed.exact_source(code)
            except Exception:
                pass
        return h, e

    def check(self, code):
        """Return (accept: bool, fp: (hash, embed)). accept=True when the candidate is a
        NOVEL architecture (or untraceable). Does NOT mutate state."""
        # RETURN ARITY IS UNCHANGED ON PURPOSE. run_seed_ga.py:565 does
        # `accept, fp = diversity_gate.check(code)`, and three GA runs are live against it.
        # The match is published on self.last_match instead of widening the tuple.
        self.last_match = None
        h, e = self._fingerprint(code)
        if h is None:
            return True, (None, None)            # can't trace -> let training judge
        if h in self.accepted:
            self.last_match = {"key": None, "cos": 1.0, "kind": "exact-clone"}
            return False, (h, e)                 # exact graph-clone
        if self.graded and e is not None and self.embeds:
            import numpy as np
            sims = [float(np.dot(e, ae)) for ae in self.embeds]
            i = int(max(range(len(sims)), key=sims.__getitem__))
            mx = sims[i]
            key = self.embed_keys[i] if i < len(self.embed_keys) else None
            # Recorded whether or not it rejects: a candidate that scraped past at 0.949 is
            # as informative to the generator as one refused at 0.951.
            self.last_match = {"key": key, "cos": mx,
                               "kind": "near-dup" if mx >= self.cos_max else "nearest"}
            if mx >= self.cos_max:
                return False, (h, e)             # near-duplicate (graded)
        return True, (h, e)

    def add(self, key, fp):
        """Add an accepted model's fingerprint to the reference set (greedy) + persist."""
        h, e = fp
        if h is None or h in self.accepted:
            return
        self.accepted.add(h)
        if self.graded and e is not None:
            self.embeds.append(e)
            self.embed_keys.append(key)          # parallel to embeds; see __init__
        self.n_accept += 1
        if self.store_path:
            try:
                import torch
                tmp = self.store_path + ".tmp"
                torch.save({"hashes": list(self.accepted), "embeds": self.embeds,
                            "embed_keys": self.embed_keys}, tmp)
                os.replace(tmp, self.store_path)
            except Exception as e2:
                self.log(f"  [diversity] persist failed: {e2}")
