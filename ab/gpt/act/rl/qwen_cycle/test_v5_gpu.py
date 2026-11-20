"""GPU test of the v5 reward worker on the workstation (real strict check, real CIFAR-10).
1. three networks that were strict-valid in seed 0 cycle 1 PPO: accuracy again close to the logged accuracy
2. a network that hangs in training is stopped by the time limit (set to 60 s for this test only)
3. a worker killed from outside is restarted and the same network is evaluated again
Usage: cd ~/nn-gpt-main && .venv/bin/python test_v5_gpu.py"""
import json, os, signal, threading, time

import qwen_cycle as Q

HANG = """import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.c = nn.Conv2d(3, 16, 3, padding=1); self.f = nn.Linear(16, 10)
    def forward(self, x):
        if torch.is_grad_enabled():
            time.sleep(1000)
        return self.f(F.adaptive_avg_pool2d(F.relu(self.c(x)), 1).flatten(1))
"""


def main():
    ok = True
    R = Q.load_reward_functions()
    path = os.path.expanduser("~/nn-gpt-main/out_qwen/final/seed0/c1_ppo/generations.jsonl")
    recs = [json.loads(l) for l in open(path)]
    valid = [r for r in recs if r["status"] == "OK" and not r["cached"] and len(r["accs"]) == 2][:3]
    w = Q.RewardWorker()
    t = time.time(); w.start(1); print(f"[TEST] worker started in {time.time() - t:.0f} s", flush=True)
    for r in valid:
        got = w.evaluate(r["code"], 2)
        old, new = 100 * sum(r["accs"]) / 2, 100 * sum(got["accs"]) / max(1, len(got["accs"]))
        good = got["stage"] == "ok" and abs(new - old) < 5
        ok &= good
        print(f"[TEST 1] gen {r['gen']}: logged {old:.1f}% -> now {new:.1f}% | train s {got['times']} | peak RAM "
              f"{got['peak_rss_gib']} GiB -> {'PASS' if good else 'FAIL'}", flush=True)
    Q.TRAIN_TIMEOUT = 60
    got = w.evaluate(HANG, 2)
    good = got["status"] == "train_timeout" and got["stage"] == "train"
    ok &= good
    print(f"[TEST 2] hanging network: {got['status']} -> {'PASS' if good else 'FAIL'}", flush=True)
    Q.TRAIN_TIMEOUT = 1800
    w.start(1)
    def killer():
        time.sleep(10); os.kill(w.p.pid, signal.SIGKILL); print("[TEST 3] (worker killed from outside)", flush=True)
    threading.Thread(target=killer, daemon=True).start()
    got = w.evaluate(valid[0]["code"], 2)
    good = got["stage"] == "ok" and got["attempts"] == 2
    ok &= good
    print(f"[TEST 3] after outside kill: {got['status']} attempts {got['attempts']} accs {got['accs']} -> {'PASS' if good else 'FAIL'}", flush=True)
    w.stop()
    print(f"[TEST] counts {w.counts}", flush=True)
    print(f"[TEST] RESULT: {'ALL TESTS PASSED' if ok else 'A TEST FAILED'}", flush=True)


if __name__ == "__main__":
    main()
