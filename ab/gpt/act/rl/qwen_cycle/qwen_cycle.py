"""One RL phase (PPO or GRPO) of the Qwen self-improvement cycle.

Generation : NNGPT vLLM sleep engine (ab.gpt.util.llm.VLLMSleep) with the trainer's LoRA adapter
Trainer    : 4-bit base + fresh LoRA (ab.gpt.act.rl.trainer_runtime)
Reward     : exact strict check and CIFAR-10 training copied from ~/nn-gpt/ab/gpt/TunePPO.py
Update     : prompt-conditioned token log-probs over the full completion, token-level clipped surrogate,
             k3 KL to the base model (adapter disabled), entropy bonus.
             PPO : advantage = r - V(prompt), value head on the last prompt token, batch-normalized.
             GRPO: Dr. GRPO advantage r - mean(group), groups with equal rewards skipped (DAPO),
                   asymmetric clipping 0.2 / 0.28 (DAPO). Truncated completions masked for both.
             Cycle 3 on (after the cycle-3 entropy collapse), GRPO runs with ENT_COEF=0 CLIP_HIGH=0.2 MASK_TRUNC=0
             BEST_WINDOW=4 STOP_ENT_FACTOR=3 (TRL GRPOTrainer conventions as in NNGPT RL.py + stage-best checkpoint).
             Cycle 4 rerun (after the cycle-4 PPO entropy collapse): PPO and GRPO both run with ENT_COEF=0 BEST_WINDOW=4
             STOP_ENT_FACTOR=3 STOP_ENT_ABS=1.2 (the relative stop alone missed a phase that started from a high-entropy model).
             An out-of-memory error in an update skips that step's update instead of failing the phase.
             v4: REWARD_MODE=scaled gives valid networks offset + scale x (accuracy - center) (default: accuracy).
             v4c: before every update, cached memory is released and the script waits until 6 GiB are free; an update
             that still runs out of memory is repeated with the identical batch (up to 3 attempts), not skipped.
             v5: each network is built and trained in a separate worker process (data in memory, no DataLoader
             worker processes), with 1800 s per training run and 12 GiB resident memory as limits (exceeded: invalid,
             -0.1 after a valid build, -0.3 during the build); a worker killed from outside is restarted and the same
             network evaluated again (shared machine). The main run waits for 6 GiB available host memory per step.
             v4: memory-bounded update (lm_head applied in 256-token chunks under checkpointing), so long
             completions no longer exhaust GPU memory; the skip above remains only as a logged safety net.
Usage      : python qwen_cycle.py   (configured by environment variables, see CFG)
"""
import ast, gc, importlib.util, json, math, os, random, re, shutil, signal, statistics as st, sys, textwrap, inspect, time
import torch
import torch.nn as nn
import torch.nn.functional as F

E = os.environ.get
HOME = os.path.expanduser("~")
CFG = dict(
    base=E("BASE", "Qwen/Qwen3.5-9B"), algo=E("ALGO", "ppo").lower(), run_dir=E("RUN_DIR", "out_qwen/test"),
    prompts=E("PROMPTS", f"{HOME}/train_prompts_v3.json"), n_prompts=int(E("N_PROMPTS", "32")),
    prompts_per_step=int(E("PROMPTS_PER_STEP", "2")), g=int(E("SAMPLES", "8")), max_steps=int(E("MAX_STEPS", "0")),
    max_new=int(E("MAX_NEW", "4096")), seed=int(E("SEED", "0")), temperature=0.8, top_p=0.9, top_k=50,
    lr=float(E("LR", "5e-5")), lora_r=16, lora_alpha=32, lora_dropout=0.05,
    kl=float(E("KL_COEF", "0.05")), ent=float(E("ENT_COEF", "0.01")), update_epochs=int(E("UPDATE_EPOCHS", "1")),
    clip_low=0.2, clip_high=float(E("CLIP_HIGH", "0.28" if E("ALGO", "ppo").lower() == "grpo" else "0.2")),
    value_coef=0.5, retrain=int(E("RETRAIN", "2")), nn_epochs=int(E("NN_EPOCHS", "10")),
    div_alpha=float(E("DIV_ALPHA", "0.2")), max_params=int(float(E("MAX_PARAMS", "1e8"))),
    # stability options (defaults = cycles 1-2 behaviour). The fixed GRPO phases (cycle 3 on) use the NNGPT/TRL
    # GRPOTrainer conventions: ENT_COEF=0, CLIP_HIGH=0.2, MASK_TRUNC=0, plus a stage-best checkpoint (BEST_WINDOW).
    mask_trunc=E("MASK_TRUNC", "1") == "1",            # 1: truncated completions excluded from the loss
    best_window=int(E("BEST_WINDOW", "0")),            # >0: keep the adapter with the best mean reward over this many steps
    stop_ent=float(E("STOP_ENT_FACTOR", "0")),         # >0: stop the phase when entropy exceeds this factor x step-1 entropy
    stop_ent_abs=float(E("STOP_ENT_ABS", "0")),        # >0: also stop when entropy exceeds this absolute value (healthy phases: 0.3-0.9)
    # reward for a strict-valid network: "acc" = mean accuracy (cycles 1-4 so far); "scaled" = offset + scale x (acc - center),
    # which makes accuracy differences between valid networks as large as the step from invalid to valid
    reward_mode=E("REWARD_MODE", "acc"), r_offset=float(E("R_OFFSET", "0.5")), r_scale=float(E("R_SCALE", "3")),
    r_center=float(E("R_CENTER", "0.6")),
)
assert CFG["algo"] in ("ppo", "grpo", "none"), CFG["algo"]
assert CFG["reward_mode"] in ("acc", "scaled"), CFG["reward_mode"]
OLD = E("OLD_CODE", f"{HOME}/nn-gpt")   # our previous code base: reward functions and AST signature are taken from here
GIB = 2 ** 30
PARK = E("PARK", "0") == "1"   # 0: trainer stays on the GPU next to the engine (no parking)


# ----------------------------------------------------------------------------- reward (exact copy)
def load_reward_functions():
    src = open(f"{OLD}/ab/gpt/TunePPO.py").read()
    tree = ast.parse(src)
    import torchvision, torchvision.transforms as transforms
    ns = {"torch": torch, "nn": nn, "F": F, "re": re, "_re": re, "textwrap": textwrap, "inspect": inspect,
          "torchvision": torchvision, "transforms": transforms, "_signal_mod": signal, "_os": os,
          "NN_TRAIN_EPOCHS": CFG["nn_epochs"], "_MAX_PARAMS": CFG["max_params"]}
    want = {"_strict_extract", "_strict_build", "_train_eval", "_gen_timeout", "_with_time_limit", "CIFARRewardEvaluator"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
            seg = ast.get_source_segment(src, node).replace("root='./data'", f"root='{OLD}/data'")
            exec(seg, ns)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_PRM" for t in node.targets):
            exec(ast.get_source_segment(src, node), ns)
    missing = want - set(ns)
    assert not missing, f"not found in TunePPO.py: {missing}"
    signal.signal(signal.SIGALRM, ns["_gen_timeout"])
    ns["strict_build"] = ns["_with_time_limit"](ns["_strict_build"])
    spec = importlib.util.spec_from_file_location("analyze_collapse", f"{OLD}/analyze_collapse.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
    ns["sig"] = ac.exact_signature
    return ns


class Reward:
    """Strict reward of one completion. v5: every network is built and trained in a separate worker process
    (RewardWorker), so a network can neither hang nor slow down the run without bound, nor exhaust the host memory.
    Same strict check, same penalties, same training (10 epochs, Adam 1e-3, batch 128, shuffled, CIFAR-10 normalized
    with mean 0.5 / std 0.5), trained RETRAIN times; cached accuracies for repeated architectures as before."""
    def __init__(self, R):
        self.R = R
        self.worker = RewardWorker()
        self.cache, self.sig_buffer = {}, []

    def __call__(self, text):
        R = self.R
        code = R["_strict_extract"](text)
        if not code or not code.strip():
            return dict(reward=-0.5, status="no_code", real=False, accs=[], sig=None, cached=False, code=code)
        sig = R["sig"](code)
        cached = bool(sig and sig in self.cache)
        w = self.worker.evaluate(code, 0 if cached else max(1, CFG["retrain"]))
        info = dict(t_train=w["times"], peak_rss_gib=w["peak_rss_gib"], worker_attempts=w["attempts"])
        if w["stage"] == "build":                                    # not strict-valid (penalty of the strict check)
            return dict(reward=w["penalty"], status=w["status"], real=False, accs=[], sig=sig, cached=False, code=code, **info)
        if w["stage"] == "train":                                    # strict-valid but training failed or hit a limit
            return dict(reward=-0.1, status=w["status"], real=False, accs=w["accs"], sig=sig, cached=False, code=code, **info)
        if cached:
            accs = self.cache[sig]
        else:
            accs = w["accs"]
            if sig:
                self.cache[sig] = accs
        acc = float(sum(accs) / len(accs))
        pen = CFG["div_alpha"] if (sig and sig in self.sig_buffer) else 0.0   # only exact repeats are penalized
        if sig:
            self.sig_buffer = (self.sig_buffer + [sig])[-50:]
        base = acc if CFG["reward_mode"] == "acc" else CFG["r_offset"] + CFG["r_scale"] * (acc - CFG["r_center"])
        return dict(reward=base - pen, accuracy=acc, penalty=pen, status="OK", real=True, accs=accs, sig=sig,
                    cached=cached, code=code, **info)


# ----------------------------------------------------------------------------- isolated reward worker (v5)
TRAIN_TIMEOUT = int(E("TRAIN_TIMEOUT", "1800"))      # s per training run (10 epochs + test pass) of one network
BUILD_HARD = int(E("BUILD_TIMEOUT", "120")) + 120    # backstop; the strict check's own 120 s alarm normally fires first
RAM_LIMIT = float(E("RAM_LIMIT_GB", "12"))           # GiB resident memory of the worker
NEED_RAM = float(E("NEED_RAM_GB", "6"))              # GiB of available host memory required before a worker starts
REWARD_DEV = E("REWARD_DEVICE", "cuda")


def host_ram_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def wait_for_host_ram(need=NEED_RAM, timeout_s=3600):
    """The machine is shared: wait until other users' jobs leave enough host memory for the reward worker."""
    t0 = time.time()
    while True:
        avail = host_ram_gib()
        if avail >= need:
            return avail
        if time.time() - t0 > timeout_s:
            print(f"[REWARD] only {avail:.1f} GiB host memory available after {timeout_s} s, starting anyway", flush=True)
            return avail
        print(f"[REWARD] {avail:.1f} GiB host memory available, waiting for {need} GiB", flush=True)
        time.sleep(30)


def load_cifar_tensors():
    """CIFAR-10 as tensors, identical to transforms.ToTensor() + Normalize((0.5,)*3, (0.5,)*3) on every image."""
    import torchvision
    out = []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR10(root=f"{OLD}/data", train=train, download=False)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().float().div(255)   # = ToTensor
        x = x.sub_(0.5).div_(0.5)                                                           # = Normalize
        out += [x, torch.tensor(ds.targets, dtype=torch.long)]
    return out


def train_eval_tensors(net, xtr, ytr, xte, yte, epochs, bs=128):
    """Same procedure as _train_eval in TunePPO.py (Adam 1e-3, cross-entropy, batch 128, reshuffled every epoch,
    last partial batch kept, then test accuracy), with the data held in memory instead of DataLoader worker processes."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            x, y = xtr[idx].to(REWARD_DEV), ytr[idx].to(REWARD_DEV)
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()
    net.eval()
    c = t = 0
    with torch.no_grad():
        for i in range(0, len(xte), bs):
            x, y = xte[i:i + bs].to(REWARD_DEV), yte[i:i + bs].to(REWARD_DEV)
            c += (net(x).argmax(1) == y).sum().item()
            t += y.size(0)
    return c / t


def _worker_main(conn, seed):
    """Reward worker process: strict check + training of one network per request. It is the first process the
    kernel kills if the machine runs out of memory (oom_score_adj 1000), so the main run survives."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    R = load_reward_functions()
    xtr, ytr, xte, yte = load_cifar_tensors()
    torch.manual_seed(seed)
    conn.send(("ready",))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        _, code, retrain = msg
        net = None
        try:
            built, status, pen = R["strict_build"](code)
        except BaseException as e:                                    # e.g. the 120 s alarm between two checks
            built, status, pen = None, f"build:{type(e).__name__}", -0.3
        if built is None:
            torch.cuda.empty_cache()
            conn.send(("built", status, pen))
            continue
        conn.send(("built", "OK", None))
        NetCls, net = built
        try:
            if retrain == 0:                                          # repeated architecture: cached accuracies
                net = None
            for k in range(retrain):
                if k > 0:
                    with torch.device(REWARD_DEV):
                        net = NetCls((1, 3, 32, 32), (10,), R["_PRM"], torch.device(REWARD_DEV))
                    net = net.to(REWARD_DEV)
                t = time.time()
                a = train_eval_tensors(net, xtr, ytr, xte, yte, CFG["nn_epochs"])
                net = None; torch.cuda.empty_cache()
                conn.send(("acc", a, time.time() - t))
            conn.send(("done",))
        except Exception as e:
            net = None; gc.collect(); torch.cuda.empty_cache()
            conn.send(("fail", f"training_failed:{type(e).__name__}"))


class RewardWorker:
    """Runs the strict check and the trainings of each network in a separate (spawned) process, with a time limit per
    training run and a resident-memory limit. A network that exceeds a limit is not strict-valid in practice and gets
    the training-failure penalty; it is logged as [REWARD] LIMIT. If the worker is killed from outside (the machine ran
    out of memory because of other jobs), the same network is evaluated again after memory is free ([REWARD] RESTART)."""
    def __init__(self):
        self.p = self.conn = None
        self.counts = dict(train_timeout=0, build_timeout=0, ram_limit=0, worker_restart=0, worker_died=0)

    def start(self, step=0):
        import multiprocessing as mp
        self.seed = CFG["seed"] * 100003 + step if step else getattr(self, "seed", CFG["seed"])
        wait_for_host_ram()
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.p = ctx.Process(target=_worker_main, args=(child, self.seed), daemon=True)
        self.p.start(); child.close()
        m, _ = self._recv(900)
        if m[0] != "ready":
            self.kill()
            raise RuntimeError(f"reward worker did not start: {m}")

    def stop(self):
        if self.p is not None and self.p.is_alive():
            try:
                self.conn.send(("stop",)); self.p.join(60)
            except (OSError, EOFError):
                pass
        if self.p is not None and self.p.is_alive():
            self.kill()
        self.p = self.conn = None

    def kill(self):
        if self.p is not None:
            self.p.kill(); self.p.join(60)
        self.p = self.conn = None

    def rss_gib(self):
        try:
            for line in open(f"/proc/{self.p.pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2 ** 20
        except (OSError, AttributeError):
            pass
        return 0.0

    def _recv(self, timeout_s, peak=0.0):
        """Next message of the worker, or ('timeout',), ('ram', rss) or ('died', exitcode); returns (message, peak rss)."""
        t0 = time.time()
        while True:
            try:
                if self.conn.poll(0.5):
                    return self.conn.recv(), peak
            except (EOFError, OSError):
                self.p.join(10)
                return ("died", self.p.exitcode), peak
            rss = self.rss_gib(); peak = max(peak, rss)
            if rss > RAM_LIMIT:
                return ("ram", rss), peak
            if not self.p.is_alive():
                return ("died", self.p.exitcode), peak
            if time.time() - t0 > timeout_s:
                return ("timeout",), peak

    def evaluate(self, code, retrain):
        """Returns dict(stage=build|train|ok, status, penalty, accs, times, peak_rss_gib, attempts)."""
        peak = 0.0
        for attempt in range(1, 4):
            if self.p is None:
                self.start()
            accs, times, stage = [], [], "build"
            self.conn.send(("eval", code, retrain))
            limit = BUILD_HARD
            while True:
                m, peak = self._recv(limit, peak)
                res = None
                if m[0] == "built":
                    if m[1] != "OK":
                        res = dict(stage="build", status=m[1], penalty=m[2])
                    else:
                        stage, limit = "train", TRAIN_TIMEOUT
                elif m[0] == "acc":
                    accs.append(m[1]); times.append(round(m[2], 1))
                elif m[0] == "done":
                    res = dict(stage="ok", status="OK", penalty=None)
                elif m[0] == "fail":
                    res = dict(stage="train", status=m[1], penalty=-0.1)
                elif m[0] in ("timeout", "ram"):
                    self.kill()
                    kind = ("train_timeout" if stage == "train" else "build_timeout") if m[0] == "timeout" else "ram_limit"
                    self.counts[kind] += 1
                    print(f"[REWARD] LIMIT {kind}: {'%.0f s' % limit if m[0] == 'timeout' else '%.1f GiB' % m[1]} "
                          f"(stage {stage}, finished runs {len(accs)})", flush=True)
                    res = dict(stage="train" if stage == "train" else "build", status=kind,
                               penalty=-0.1 if stage == "train" else -0.3)
                elif m[0] == "died":
                    self.p = self.conn = None
                    self.counts["worker_restart"] += 1
                    print(f"[REWARD] RESTART: worker ended with exit code {m[1]} (peak {peak:.1f} GiB, host memory "
                          f"available {host_ram_gib():.1f} GiB) - evaluating the same network again (attempt {attempt})", flush=True)
                    break
                if res is not None:
                    res.update(accs=accs, times=times, peak_rss_gib=round(peak, 2), attempts=attempt)
                    return res
        self.counts["worker_died"] += 1
        print("[REWARD] LIMIT worker_died: the worker was killed in 3 attempts on this network", flush=True)
        return dict(stage="train" if stage == "train" else "build", status="worker_died",
                    penalty=-0.1 if stage == "train" else -0.3, accs=[], times=[], peak_rss_gib=round(peak, 2), attempts=3)


# ----------------------------------------------------------------------------- statistics + plot
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def mean_ci(x):
    if not x:
        return None, None, None
    m = st.mean(x)
    if len(x) < 2:
        return m, m, m
    try:
        from scipy.stats import t
        tc = float(t.ppf(0.975, len(x) - 1))
    except Exception:
        tc = 1.96
    h = tc * st.stdev(x) / math.sqrt(len(x))
    return m, m - h, m + h


def plot_steps(rows, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BL, OR, INK, MU, GR = "#2a78d6", "#eb6834", "#1f1f1e", "#6f6e69", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MU, "xtick.color": MU, "ytick.color": MU,
                         "axes.spines.top": False, "axes.spines.right": False})
    x = [r["step"] for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    ax[0].fill_between(x, [r["valid_lo"] for r in rows], [r["valid_hi"] for r in rows], color=BL, alpha=0.15, lw=0)
    ax[0].plot(x, [r["valid_pct"] for r in rows], color=BL, lw=2, marker="o", ms=4, label="strict-valid networks")
    for r in rows:
        ax[0].annotate(f"{r['valid']}/{r['n']}", (r["step"], r["valid_pct"]), textcoords="offset points",
                       xytext=(0, 5), ha="center", fontsize=6.5, color=INK)
    ax[0].set_ylim(0, 100); ax[0].set_ylabel("strict-valid (%)")
    ax[0].set_title("Strict-valid generation rate per update step (Wilson 95% CI)", fontsize=8, color=INK, loc="left")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    xs = [r["step"] for r in rows if r["acc_mean"] is not None]
    if xs:
        a = [r for r in rows if r["acc_mean"] is not None]
        ax[1].fill_between(xs, [r["acc_lo"] for r in a], [r["acc_hi"] for r in a], color=OR, alpha=0.15, lw=0)
        ax[1].plot(xs, [r["acc_mean"] for r in a], color=OR, lw=2, marker="o", ms=4, label="mean of valid networks (95% CI)")
        ax[1].plot(xs, [r["acc_best"] for r in a], color=OR, lw=1.2, ls="--", marker="s", ms=3.5, label="best network")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("CIFAR-10 accuracy (%)"); ax[1].set_xlabel("update step")
    ax[1].set_title("Accuracy of strict-valid networks, trained as written", fontsize=8, color=INK, loc="left")
    ax[1].legend(frameon=False, fontsize=7, loc="lower left")
    for a_ in ax:
        a_.grid(axis="y", color=GR, zorder=0)
    fig.suptitle(title, fontsize=8.5, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------- policy statistics
def wait_for_memory(step, attempt, need_gib=6.0, timeout_s=900):
    """Before an update: release cached memory and wait until the shared GPU
    has `need_gib` free (the vLLM engine of the step must be gone, other users' jobs may hold memory)."""
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free >= need_gib or time.time() - t0 > timeout_s:
            if free < need_gib:
                print(f"[CYCLE] step {step}: only {free:.1f} GiB free after {timeout_s} s, trying the update anyway", flush=True)
            return free
        print(f"[CYCLE] step {step} (attempt {attempt}): {free:.1f} GiB free, waiting for {need_gib} GiB", flush=True)
        time.sleep(15); gc.collect(); torch.cuda.empty_cache()


def completion_stats(model, prompt_ids, comp_ids, chunk=256):
    """Per-token log-prob and entropy of the completion, conditioned on the prompt.
    Memory-bounded: the backbone returns only the last hidden states, and the vocabulary projection (lm_head) is
    applied in chunks of `chunk` tokens inside activation checkpointing, so the full [T x vocabulary] logit matrix and
    its gradient are never held in memory (they were the cause of the out-of-memory errors on long completions)."""
    ids = torch.tensor(prompt_ids + comp_ids, device="cuda").unsqueeze(0)
    T = len(comp_ids)
    tgt = torch.tensor(comp_ids, device="cuda")
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone, head = getattr(inner, "model", None), inner.get_output_embeddings()
    hidden = backbone(input_ids=ids).last_hidden_state[0, -(T + 1):-1]   # positions that predict the T completion tokens

    def part(h, y):
        l = head(h).float()
        lse = torch.logsumexp(l, -1)
        return l.gather(-1, y.unsqueeze(-1)).squeeze(-1) - lse, lse - (torch.softmax(l, -1) * l).sum(-1)

    lps, ents = [], []
    for s in range(0, T, chunk):
        lp, en = torch.utils.checkpoint.checkpoint(part, hidden[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
        lps.append(lp); ents.append(en)
    return torch.cat(lps), torch.cat(ents)


def completion_stats_reference(model, prompt_ids, comp_ids, chunk=512):
    """The previous implementation (full logits); kept only to verify that the memory-bounded version is identical."""
    ids = torch.tensor(prompt_ids + comp_ids, device="cuda").unsqueeze(0)
    T = len(comp_ids)
    logits = model(input_ids=ids, logits_to_keep=T + 1).logits[0, :-1]
    tgt = torch.tensor(comp_ids, device="cuda")

    def part(l, y):
        l = l.float()
        lse = torch.logsumexp(l, -1)
        return l.gather(-1, y.unsqueeze(-1)).squeeze(-1) - lse, lse - (torch.softmax(l, -1) * l).sum(-1)

    lps, ents = [], []
    for s in range(0, T, chunk):
        lp, en = torch.utils.checkpoint.checkpoint(part, logits[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
        lps.append(lp); ents.append(en)
    return torch.cat(lps), torch.cat(ents)


class ValueHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, h):
        return self.head(h.float()).squeeze(-1)


@torch.no_grad()
def prompt_hidden(model, prompt_ids):
    out = model(input_ids=torch.tensor(prompt_ids, device="cuda").unsqueeze(0), output_hidden_states=True, logits_to_keep=1)
    return out.hidden_states[-1][0, -1].detach()


def update(model, vhead, opt, batch):
    """batch: list of dicts with pids, cids, reward, group, truncated. Returns a summary dict."""
    algo = CFG["algo"]
    info = dict(n_batch=len(batch), n_used=0, skipped_equal_groups=0, masked_truncated=0)
    if algo == "none":
        return info
    rewards = torch.tensor([s["reward"] for s in batch], device="cuda", dtype=torch.float32)
    keep = [not (s["truncated"] and CFG["mask_trunc"]) for s in batch]
    info["masked_truncated"] = sum(not k for k in keep)
    adv = torch.zeros_like(rewards)
    hid = {}
    if algo == "grpo":
        groups = {}
        for i, s in enumerate(batch):
            groups.setdefault(s["group"], []).append(i)
        for idx in groups.values():
            r = rewards[idx]
            if len(idx) < 2 or float(r.max() - r.min()) < 1e-6:       # equal rewards: no information (DAPO)
                for i in idx:
                    keep[i] = False
                info["skipped_equal_groups"] += 1
                continue
            adv[idx] = r - r.mean()                                      # Dr. GRPO: no std normalization
    else:
        for s in batch:
            key = tuple(s["pids"])
            if key not in hid:
                hid[key] = prompt_hidden(model, s["pids"])
        with torch.no_grad():
            v = torch.stack([vhead(hid[tuple(s["pids"])]) for s in batch])
        adv = rewards - v
        if len(adv) >= 4 and float(adv.std()) > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    use = [i for i, k in enumerate(keep) if k and len(batch[i]["cids"]) >= 1]
    info["n_used"] = len(use)
    if not use and algo != "ppo":
        return info
    # reference (adapter off) and old log-probs, no gradient
    model.eval()
    with torch.no_grad():
        ref = {}
        if CFG["kl"] > 0:
            with model.disable_adapter():
                for i in use:
                    ref[i] = completion_stats(model, batch[i]["pids"], batch[i]["cids"])[0]
        old = {}
        if CFG["update_epochs"] > 1:
            for i in use:
                old[i] = completion_stats(model, batch[i]["pids"], batch[i]["cids"])[0]
    total_tokens = sum(len(batch[i]["cids"]) for i in use) or 1
    model.train()
    agg = dict(policy_loss=0.0, kl=0.0, entropy=0.0, value_loss=0.0, clip_frac=0.0)
    for ep in range(CFG["update_epochs"]):
        opt.zero_grad(set_to_none=True)
        for i in use:
            s = batch[i]
            lp, en = completion_stats(model, s["pids"], s["cids"])
            o = old[i] if i in old else lp.detach()
            ratio = torch.exp(lp - o)
            a = adv[i]
            s1 = ratio * a
            s2 = torch.clamp(ratio, 1 - CFG["clip_low"], 1 + CFG["clip_high"]) * a
            tok = -torch.min(s1, s2)
            if CFG["kl"] > 0:
                d = ref[i] - lp
                kl = torch.exp(d) - d - 1
                tok = tok + CFG["kl"] * kl
                agg["kl"] += float(kl.sum()) / total_tokens
            if CFG["ent"] > 0:
                tok = tok - CFG["ent"] * en
            (tok.sum() / total_tokens).backward()                         # token-level aggregation (DAPO)
            agg["policy_loss"] += float(-torch.min(s1, s2).sum()) / total_tokens
            agg["entropy"] += float(en.sum()) / total_tokens
            agg["clip_frac"] += float(((ratio < 1 - CFG["clip_low"]) | (ratio > 1 + CFG["clip_high"])).float().sum()) / total_tokens
            del lp, en, ratio, s1, s2, tok
            torch.cuda.empty_cache()
        if algo == "ppo":
            vs = torch.stack([vhead(hid[tuple(s["pids"])]) for s in batch])
            vloss = F.mse_loss(vs, rewards)
            (CFG["value_coef"] * vloss).backward()
            agg["value_loss"] += float(vloss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        torch.nn.utils.clip_grad_norm_(vhead.parameters(), 1.0)
        opt.step()
        torch.cuda.empty_cache()
    for k in agg:
        agg[k] /= CFG["update_epochs"]
    info.update(agg)
    return info


def build_engine_with_retry(gen, adapter, model, tries=60):
    """Size the vLLM budget to the memory that is actually free (the GPU is shared) and retry instead of crashing."""
    for attempt in range(1, tries + 1):
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        util = min(0.55, (free - 1.0 * GIB) / total)
        if util < 0.40:
            print(f"[CYCLE] only {free / GIB:.1f} GiB free on the GPU (other users?) - waiting 60 s "
                  f"(attempt {attempt}/{tries})", flush=True)
            time.sleep(60)
            continue
        os.environ["NNGPT_VLLM_GPU_MEMORY_UTILIZATION"] = f"{util:.3f}"
        gen._build_failed = False                      # a failed build would otherwise disable vLLM for good
        gen.prepare(adapter, model if PARK else None)
        if gen.engine is not None:
            return gen.engine
        print(f"[CYCLE] engine build failed (budget {util:.2f}) - retry in 60 s (attempt {attempt}/{tries})", flush=True)
        time.sleep(60)
    sys.exit("[CYCLE] FATAL: vLLM engine could not be built")


# ----------------------------------------------------------------------------- main loop
def main():
    run = CFG["run_dir"]; os.makedirs(run, exist_ok=True)
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    from ab.gpt.act.rl.trainer_runtime import (load_tokenizer, load_quantized_causal_lm, build_lora_config,
                                               attach_or_resume_lora, enable_non_reentrant_gradient_checkpointing)
    import ab.gpt.util.llm.VLLMSleep as V
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = json.load(open(CFG["prompts"]))[:CFG["n_prompts"]]
    json.dump(dict(CFG, prompt_texts=prompts), open(f"{run}/config.json", "w"), indent=1)
    print("[CYCLE] config: " + " ".join(f"{k}={v}" for k, v in CFG.items()), flush=True)

    R = load_reward_functions()
    reward = Reward(R)
    tok = load_tokenizer(CFG["base"])
    prec = {"dtype": torch.bfloat16, "bf16": True, "fp16": False, "label": "bf16"}
    model = load_quantized_causal_lm(model_source=CFG["base"], precision=prec, train_device="cuda", use_deepspeed=False)
    model = attach_or_resume_lora(model, peft_config=build_lora_config(r=CFG["lora_r"], alpha=CFG["lora_alpha"],
                                  dropout=CFG["lora_dropout"]), stage_adapter_dir=None, log_prefix="[CYCLE]")
    model.print_trainable_parameters()
    enable_non_reentrant_gradient_checkpointing(model, log_prefix="[CYCLE]")
    cfg_m = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    vhead = ValueHead(cfg_m.hidden_size).cuda()
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": CFG["lr"]},
                             {"params": vhead.parameters(), "lr": CFG["lr"] * 10}], weight_decay=0.01)

    gen = V.VLLMGeneration(CFG["base"], tok, 8192, temperature=CFG["temperature"], top_k=CFG["top_k"], top_p=CFG["top_p"])
    gen.suppress_thinking = True
    log_gen, log_step = open(f"{run}/generations.jsonl", "a"), f"{run}/steps.jsonl"
    rows, gen_count, group = [], 0, 0
    best, ent0 = None, None
    steps = [prompts[i:i + CFG["prompts_per_step"]] for i in range(0, len(prompts), CFG["prompts_per_step"])]
    if CFG["max_steps"]:
        steps = steps[:CFG["max_steps"]]
    for si, step_prompts in enumerate(steps, 1):
        t0 = time.time()
        adapter = os.path.abspath(f"{run}/adapter_step")
        model.save_pretrained(adapter)
        eng = build_engine_with_retry(gen, adapter, model)
        eng.suppress_thinking = True
        pid_list = [eng._token_ids(p) for p in step_prompts]
        sp = SamplingParams(n=CFG["g"], temperature=CFG["temperature"], top_p=CFG["top_p"], top_k=CFG["top_k"],
                            max_tokens=CFG["max_new"], seed=CFG["seed"] * 100003 + si)
        outs = eng.llm.generate([TokensPrompt(prompt_token_ids=p) for p in pid_list], sp, lora_request=eng._lora_request())
        gen.release(model if PARK else None)
        t_gen = time.time() - t0
        batch, step_recs = [], []
        c0 = dict(reward.worker.counts)
        reward.worker.start(si)                   # fresh reward worker for this step (stopped before the update)
        for p_text, pids, out in zip(step_prompts, pid_list, outs):
            group += 1
            for o in out.outputs:
                gen_count += 1
                r = reward(o.text)
                s = dict(pids=pids, cids=list(o.token_ids), reward=r["reward"], group=group,
                         truncated=(o.finish_reason == "length"))
                batch.append(s)
                rec = dict(gen=gen_count, step=si, group=group, algo=CFG["algo"], status=r["status"], real=r["real"],
                           accuracy=r.get("accuracy"), accs=r["accs"], cached=r["cached"], sig=r["sig"],
                           penalty=r.get("penalty", 0.0), reward=r["reward"], n_tokens=len(o.token_ids),
                           finish=o.finish_reason, prompt=p_text, code=r["code"], raw_output=o.text, time=time.time(),
                           t_train=r.get("t_train"), peak_rss_gib=r.get("peak_rss_gib"), worker_attempts=r.get("worker_attempts"))
                log_gen.write(json.dumps(rec) + "\n"); log_gen.flush()
                acc_txt = f" acc={100 * r['accuracy']:.1f}%" if r["real"] else ""
                print(f"  [{si}/{len(steps)}] gen {gen_count}: {r['status']}{acc_txt}"
                      f" reward={r['reward']:.3f} tokens={len(o.token_ids)} {o.finish_reason}"
                      f" train_s={r.get('t_train')} rss={r.get('peak_rss_gib')}", flush=True)
                step_recs.append(rec)
        reward.worker.stop()
        limits = {k: reward.worker.counts[k] - c0[k] for k in c0}
        t_rew = time.time() - t0 - t_gen
        info = None
        for attempt in range(1, 4):
            free_gib = wait_for_memory(si, attempt)
            try:
                info = update(model, vhead, opt, batch)
                info["free_gib_before_update"] = round(free_gib, 2); info["update_attempts"] = attempt
                break
            except torch.OutOfMemoryError as e:
                # the GPU is shared: free everything, wait for memory and repeat the SAME update (identical batch);
                # nothing is skipped unless three attempts fail, which is logged and must not happen in reported runs
                print(f"[CYCLE] OOM in update of step {si}, attempt {attempt}: {str(e)[:400]}", flush=True)
                opt.zero_grad(set_to_none=True); vhead.zero_grad(set_to_none=True)
                model.train(); gc.collect(); torch.cuda.empty_cache()
        if info is None:
            print(f"[CYCLE] OOM in update of step {si}: update skipped after 3 attempts", flush=True)
            info = dict(n_batch=len(batch), n_used=0, skipped_equal_groups=0, masked_truncated=0, oom_skipped=1)
        if info.get("oom_skipped"):
            opt.zero_grad(set_to_none=True); vhead.zero_grad(set_to_none=True)
            model.train(); torch.cuda.empty_cache()
        t_upd = time.time() - t0 - t_gen - t_rew
        accs = [100 * r["accuracy"] for r in step_recs if r["real"]]
        k, n = len(accs), len(batch)
        vp, vlo, vhi = wilson(k, n)
        am, alo, ahi = mean_ci(accs)
        groups_inf = 0
        for gid in {s["group"] for s in batch}:
            sigs = {r["sig"] for r in step_recs if r["group"] == gid and r["real"]}
            groups_inf += len(sigs - {None}) >= 2
        row = dict(step=si, n=n, valid=k, valid_pct=vp, valid_lo=vlo, valid_hi=vhi, acc_mean=am, acc_lo=alo,
                   acc_hi=ahi, acc_best=max(accs) if accs else None, informative_groups=groups_inf,
                   truncated=sum(s["truncated"] for s in batch), mean_reward=st.mean(s["reward"] for s in batch),
                   t_gen=t_gen, t_reward=t_rew, t_update=t_upd, reward_limits=limits, **info)
        rows.append(row)
        with open(log_step, "a") as f:
            f.write(json.dumps(row) + "\n")
        plot_steps(rows, f"{run}/epoch_statistics.png",
                   f"Qwen3.5-9B, {CFG['algo'].upper()} phase, base {os.path.basename(CFG['base'].rstrip('/'))}: "
                   f"{CFG['g']} samples x {CFG['prompts_per_step']} prompts per step")
        print(f"[CYCLE] step {si}/{len(steps)}: valid {k}/{n} ({vp:.0f}% [{vlo:.0f},{vhi:.0f}]) "
              f"acc {'-' if am is None else f'{am:.1f}'} best {'-' if not accs else f'{max(accs):.1f}'} | "
              f"used {info['n_used']} skipped_groups {info['skipped_equal_groups']} masked {info['masked_truncated']} "
              f"kl {info.get('kl', 0):.4f} ent {info.get('entropy', 0):.3f} | "
              f"time gen {t_gen / 60:.1f} reward {t_rew / 60:.1f} update {t_upd / 60:.1f} min", flush=True)
        if CFG["best_window"]:
            # stage-best checkpoint: the rewards of step si were earned by the adapter saved at the start of step si
            score = st.mean(r["mean_reward"] for r in rows[-CFG["best_window"]:])
            if best is None or score > best[0]:
                best = (score, si)
                shutil.rmtree(f"{run}/adapter_best", ignore_errors=True)
                shutil.copytree(adapter, f"{run}/adapter_best")
                json.dump(dict(step=si, window=CFG["best_window"], mean_reward=score), open(f"{run}/best.json", "w"))
                print(f"[CYCLE] best checkpoint -> adapter of step {si} (mean reward over last "
                      f"{min(si, CFG['best_window'])} steps {score:.3f})", flush=True)
        if ent0 is None and info.get("entropy"):
            ent0 = info["entropy"]                                        # reference: first update of the phase
        if CFG["stop_ent"] and ent0 and info.get("entropy", 0) > CFG["stop_ent"] * ent0:
            print(f"[CYCLE] STOP: entropy {info['entropy']:.3f} > {CFG['stop_ent']} x step-1 entropy {ent0:.3f}", flush=True)
            break
        if CFG["stop_ent_abs"] and info.get("entropy", 0) > CFG["stop_ent_abs"]:
            print(f"[CYCLE] STOP: entropy {info['entropy']:.3f} > absolute limit {CFG['stop_ent_abs']}", flush=True)
            break
    if CFG["best_window"]:
        model.save_pretrained(f"{run}/adapter_last")
        shutil.rmtree(f"{run}/adapter", ignore_errors=True)
        shutil.copytree(f"{run}/adapter_best", f"{run}/adapter")   # the merge uses the stage-best adapter
        print(f"[CYCLE] phase adapter = best checkpoint {json.load(open(f'{run}/best.json'))}", flush=True)
    else:
        model.save_pretrained(f"{run}/adapter")
    torch.save(vhead.state_dict(), f"{run}/value_head.pt")
    print("[CYCLE] reward limits in this phase: " + " ".join(f"{k}={v}" for k, v in reward.worker.counts.items()), flush=True)
    open(f"{run}/DONE", "w").write(time.ctime())
    gen.shutdown()
    print("[CYCLE] PHASE DONE", flush=True)


if __name__ == "__main__":
    main()
