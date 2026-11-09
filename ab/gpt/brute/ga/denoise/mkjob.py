"""Emit a K8s Indexed Job manifest for one stage of the parallel evolve loop.

The coordinator needs to launch three differently-shaped jobs per generation (generate,
screen, train) that differ only in their command and shard count. Templating them here keeps
the pod spec -- uid/gid, the /a/mm hostPath, the pod-local /scratch emptyDir, the GPU
request, LEMUR_CKPT_DIR -- in ONE place. Writing it inline in the shell script would mean
three copies of a spec whose details have already cost us a day each time one drifted.

completionMode: Indexed is what gives each pod its JOB_COMPLETION_INDEX, which every stage
uses as its --shard.
"""
import argparse
import json

IMAGE = "abrainone/ai-linux:cu12.6.3-latest"
HOSTPATH = "/shared/ssd/home/b-b-ahmed/thesis"


def build(name, completions, command, gpus, cpu, mem, scratch):
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name},
        "spec": {
            "completions": completions,
            "parallelism": completions,
            "completionMode": "Indexed",
            # backoffLimit 0: a shard that dies should surface as a failure the coordinator
            # sees, not be silently retried into a half-finished generation.
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {"spec": {
                "restartPolicy": "Never",
                "containers": [{
                    "name": "ml",
                    "image": IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-c"],
                    "args": [command],
                    "env": [
                        {"name": "LEMUR_CKPT_DIR", "value": "/a/mm/sweep_out/ckpt"},
                        {"name": "HF_HOME", "value": "/a/mm/hf_cache"},
                        {"name": "HF_HUB_DISABLE_TELEMETRY", "value": "1"},
                    ],
                    "securityContext": {"allowPrivilegeEscalation": False,
                                        "runAsUser": 1067, "runAsGroup": 1376},
                    "volumeMounts": [
                        {"mountPath": "/dev/shm", "name": "shm"},
                        {"mountPath": "/a/mm", "name": "v0"},
                        {"mountPath": "/scratch", "name": "scratch"},
                    ],
                    "resources": {
                        "requests": {"cpu": cpu, "memory": mem,
                                     **({"nvidia.com/gpu": gpus} if gpus else {})},
                        "limits": {"cpu": cpu * 2, "memory": mem,
                                   **({"nvidia.com/gpu": gpus} if gpus else {})},
                    },
                }],
                "volumes": [
                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
                    {"name": "v0", "hostPath": {"path": HOSTPATH, "type": "Directory"}},
                    {"name": "scratch", "emptyDir": {"sizeLimit": scratch}},
                ],
            }},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--completions", type=int, required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--mem", default="48Gi")
    ap.add_argument("--scratch", default="30Gi")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = build(a.name, a.completions, a.command, a.gpus, a.cpu, a.mem, a.scratch)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"wrote {a.out}  ({a.name}: {a.completions} shards, {a.gpus} GPU each)")


if __name__ == "__main__":
    raise SystemExit(main())
