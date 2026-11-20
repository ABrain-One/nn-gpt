"""Strict check of generated networks: the same check, statuses and output files as check_strict.py,
but every network is built in a separate worker process. A network that exhausts host RAM or hangs
kills only the worker: it is recorded as 'killed:memory' or 'killed:timeout' (invalid, not OK)
and a fresh worker continues with the next network.
Usage (old env, from ~/nn-gpt): python check_strict_safe.py run0.jsonl run1.jsonl ..."""
import collections, json, os, re, select, subprocess, sys, time

RSS_LIMIT_GB = float(os.environ.get("RSS_LIMIT_GB", "16"))   # a normal build needs < 3 GB host RAM
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "300"))         # backstop; the check itself has a 120 s alarm
SIG4 = r"def __init__\(self,\s*in_shape,\s*out_shape,\s*prm,\s*device"

WORKER = r'''
import ast, json, os, re, signal, sys, torch, torch.nn as nn, torch.nn.functional as F
out = os.fdopen(os.dup(1), "w"); os.dup2(2, 1)       # prints from generated code go to stderr, not the reply channel
try: open("/proc/self/oom_score_adj", "w").write("1000")   # if RAM runs out, the kernel kills this worker first
except OSError: pass
src = open(os.path.expanduser("~/nn-gpt/ab/gpt/TunePPO.py")).read(); tree = ast.parse(src)
ns = {"torch": torch, "nn": nn, "F": F, "_re": re, "_signal_mod": signal, "_os": os, "_MAX_PARAMS": int(1e8)}
for node in tree.body:   # copy the exact check functions from TunePPO.py (as check_strict.py)
    if isinstance(node, ast.FunctionDef) and node.name in ("_strict_extract", "_strict_build", "_gen_timeout", "_with_time_limit"):
        exec(ast.get_source_segment(src, node), ns)
    elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_PRM" for t in node.targets):
        exec(ast.get_source_segment(src, node), ns)
signal.signal(signal.SIGALRM, ns["_gen_timeout"])
build = ns["_with_time_limit"](ns["_strict_build"])
out.write("READY\n"); out.flush()
while True:
    line = sys.stdin.readline()
    if not line: break
    code = ns["_strict_extract"](json.loads(line))
    sig4 = bool(re.search(SIG4, code))
    res, status, pen = build(code)
    ok = res is not None
    del res; torch.cuda.empty_cache()
    out.write(json.dumps({"status": status, "ok": ok, "sig4": sig4}) + "\n"); out.flush()
'''.replace("SIG4", repr(SIG4))


def rss_gb(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 0.0


class Worker:
    def __init__(self):
        self.p = None

    def start(self):
        self.p = subprocess.Popen([sys.executable, "-c", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.buf = b""
        if self._reply() != "READY":
            sys.exit("[SAFE] FATAL: worker did not start")

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.kill()
        if self.p:
            self.p.wait()
        self.p = None

    def _reply(self):
        """Next reply line, or a 'killed:*' status if the worker dies, exceeds the RAM limit or hangs."""
        t0, fd = time.time(), self.p.stdout.fileno()
        while True:
            if b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                return "READY" if line == b"READY" else json.loads(line)
            if rss_gb(self.p.pid) > RSS_LIMIT_GB:
                self.stop(); return {"status": "killed:memory"}
            if time.time() - t0 > TIMEOUT_S:
                self.stop(); return {"status": "killed:timeout"}
            if select.select([fd], [], [], 0.2)[0]:
                chunk = os.read(fd, 65536)
                if chunk:
                    self.buf += chunk; continue
                rc = self.p.wait(); self.p = None          # worker died (e.g. the kernel's out-of-memory killer)
                return {"status": "killed:memory" if rc == -9 else f"killed:exit{rc}"}

    def check(self, text):
        if self.p is None:
            self.start()
        self.p.stdin.write((json.dumps(text) + "\n").encode()); self.p.stdin.flush()
        rep = self._reply()
        if rep["status"].startswith("killed"):
            self.stop()
            rep.update(ok=False, sig4=bool(re.search(SIG4, text)))
        return rep


def main():
    w = Worker()
    for f in sys.argv[1:]:
        R = [json.loads(l) for l in open(f)]; st = collections.Counter(); ok = sig4 = 0
        for i, r in enumerate(R):
            rep = w.check(r["text"])
            r["strict"] = rep["status"]
            st[rep["status"]] += 1; ok += rep["ok"]; sig4 += rep["sig4"]
            if rep["status"].startswith("killed"):
                print(f"[SAFE] {os.path.basename(f)} idx {r.get('idx', i)}: {rep['status']}", file=sys.stderr, flush=True)
            if (i + 1) % 25 == 0:
                print(f"[SAFE] {os.path.basename(f)}: {i + 1}/{len(R)} checked, OK so far {ok}", file=sys.stderr, flush=True)
        toks = sorted(r["n_tokens"] for r in R)
        print(f"{os.path.basename(f)}: strict OK {ok}/{len(R)} | stages {dict(st.most_common())} | "
              f"NN-Dataset __init__ in code {sig4}/{len(R)} | tokens median {toks[len(toks)//2]} max {toks[-1]} | "
              f"hit 2048: {sum(t >= 2048 for t in toks)}", flush=True)
        with open(f.replace(".jsonl", "_strict.jsonl"), "w") as g:
            for r in R:
                g.write(json.dumps(r) + "\n")
    w.stop()


if __name__ == "__main__":
    main()
