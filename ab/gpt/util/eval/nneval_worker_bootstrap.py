import argparse
import faulthandler
import os
import sys
import threading
import time
from multiprocessing.connection import Connection


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return int(default)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _make_preferred_oom_victim() -> None:
    """Tell the kernel to kill this worker first when the host runs out of memory.

    This is the single change that keeps the run alive. Without it the OOM
    killer scores processes by how much memory they hold, so the process it
    picks is the trainer -- it holds the LLM and has by far the largest
    resident set -- even though it was the evaluation of a generated network
    that exhausted the memory. The trainer dies, its terminal session goes
    with it, and nothing in Python ever sees an exception, because SIGKILL
    cannot be caught. Raising this worker's oom_score_adj inverts that choice:
    the worker is killed, the pool observes a closed pipe, and the model is
    recorded as a failed evaluation.

    Only raising the value is possible without privileges, which is all that
    is needed here.

    Setting an absolute value is not enough. oom_score_adj is inherited, and
    a desktop session hands its applications a non-zero one -- a run started
    from the IDE arrives here already at 200 -- so the worker has to be raised
    above whatever it inherited rather than to a fixed number that a future
    baseline could meet or exceed.
    """
    try:
        with open("/proc/self/oom_score_adj", "r", encoding="utf-8") as handle:
            inherited = int(handle.read().strip() or 0)
    except (OSError, ValueError):
        inherited = 0

    override = os.environ.get("NNGPT_NNEVAL_WORKER_OOM_SCORE_ADJ", "").strip()
    if override:
        adjustment = _int_env("NNGPT_NNEVAL_WORKER_OOM_SCORE_ADJ", 800)
    else:
        delta = max(1, _int_env("NNGPT_NNEVAL_WORKER_OOM_SCORE_DELTA", 600))
        adjustment = inherited + delta
    # Never below what was inherited: that would make this worker a *better*
    # survivor than the trainer, which is the failure being fixed.
    adjustment = max(inherited + 1, min(1000, adjustment))

    try:
        with open("/proc/self/oom_score_adj", "w", encoding="utf-8") as handle:
            handle.write(str(adjustment))
    except OSError:
        return
    print(
        f"[NNEval Worker] oom_score_adj {inherited} -> {adjustment} "
        f"(pid={os.getpid()}): this worker is now the kernel's preferred victim",
        file=sys.stderr,
        flush=True,
    )


def _die_with_parent() -> None:
    """Ask the kernel to kill this worker if the trainer disappears.

    The worker runs in its own session so that signals aimed at the trainer's
    process group cannot reach it, which otherwise leaves it orphaned and
    holding GPU memory when the trainer exits.
    """
    try:
        import ctypes
        import signal

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0
        )
    except Exception:
        pass


def _limit_address_space() -> None:
    """Optional hard ceiling on this worker's address space.

    Off by default and deliberately so: CUDA reserves an enormous virtual
    address range for unified addressing, so an RLIMIT_AS that sounds generous
    will stop a GPU worker from initialising at all. It is worth setting only
    for CPU evaluation, where it converts a runaway allocation into a
    MemoryError this worker can report instead of a kill the trainer absorbs.
    """
    limit_gib = _float_env("NNGPT_NNEVAL_WORKER_AS_LIMIT_GIB", 0.0)
    if limit_gib <= 0:
        return
    try:
        import resource

        limit_bytes = int(limit_gib * (1024 ** 3))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit_bytes = min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    except (ImportError, OSError, ValueError):
        pass


WORKER_MEMORY_ABORT_EXIT_CODE = 17
# A second code for the host-floor abort. The parent has to tell the two apart:
# an abort for this worker's own size is the worker's fault and retrying on a
# fresh one helps, while an abort because the host is short of memory may not be
# the worker's fault at all -- if something else is holding the memory, every
# retry aborts the same way and the run makes no progress.
WORKER_HOST_FLOOR_EXIT_CODE = 18


def _start_memory_guard() -> None:
    """Abort this worker before the machine runs out of memory.

    oom_score_adj decides *who* the kernel kills, but not *whether* it has to
    kill. On a 31 GiB machine the trainer holds ~14 GiB and an evaluation
    worker was observed holding 12 GiB, which leaves the whole system a few
    hundred MiB of headroom; at that point the kernel kills something, and
    anything sharing the terminal -- the IDE that started the run, the shell --
    can go with it. No in-process handler can prevent that, because SIGKILL is
    not deliverable to Python.

    So the worker polices itself. A daemon thread watches its own resident set
    and the host's available memory, and the moment either crosses the line it
    exits immediately with a distinctive status. The parent sees the pipe close,
    marks that one model as failed, and continues -- the same recovery path a
    kernel kill would take, except it happens while the machine still has room
    to breathe, so nothing else is ever at risk.

    os._exit is deliberate: it skips interpreter teardown, which can itself
    allocate, and cannot be swallowed by an except: in library code. Freeing
    the memory is the kernel's job once the process is gone.
    """
    # The host floor is the real protection: it targets the dangerous condition
    # itself -- the machine running out of memory -- no matter which process is
    # holding what. The RSS cap is only a secondary catch for a single runaway
    # model, so its default sits ABOVE normal usage on purpose. Observed steady
    # state for a CIFAR-10 evaluation on the reference machine is 11-12 GiB, so
    # a cap at or below that would abort healthy evaluations and turn a crash
    # into a run of failures, which is worse.
    rss_limit_gib = _float_env("NNGPT_NNEVAL_WORKER_RSS_ABORT_GIB", 14.0)
    host_floor_gib = _float_env("NNGPT_NNEVAL_HOST_ABORT_GIB", 3.0)
    interval = max(0.1, _float_env("NNGPT_NNEVAL_MEMORY_GUARD_INTERVAL", 0.5))
    if rss_limit_gib <= 0 and host_floor_gib <= 0:
        return

    def _meminfo_available_gib():
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / (1024.0 * 1024.0)
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _self_rss_gib():
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / (1024.0 * 1024.0)
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _watch():
        while True:
            time.sleep(interval)
            rss = _self_rss_gib()
            avail = _meminfo_available_gib()
            reason = None
            host_floor_hit = False
            if rss_limit_gib > 0 and rss is not None and rss >= rss_limit_gib:
                reason = f"own RSS {rss:.2f} GiB reached limit {rss_limit_gib:.2f} GiB"
            elif host_floor_gib > 0 and avail is not None and avail <= host_floor_gib:
                host_floor_hit = True
                reason = (
                    f"host MemAvailable {avail:.2f} GiB fell to floor "
                    f"{host_floor_gib:.2f} GiB (this worker holds "
                    f"{rss if rss is not None else float('nan'):.2f} GiB)"
                )
            if reason is None:
                continue
            try:
                print(
                    f"[NNEval Worker] MEMORY GUARD pid={os.getpid()}: {reason}. "
                    "Aborting this evaluation so the trainer and the terminal survive.",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
            os._exit(
                WORKER_HOST_FLOOR_EXIT_CODE if host_floor_hit
                else WORKER_MEMORY_ABORT_EXIT_CODE
            )

    threading.Thread(target=_watch, name="nngpt-memory-guard", daemon=True).start()


def _harden_worker_process() -> None:
    """Apply every protection that must be in place before torch is imported.

    These run first because a CUDA context, once created, cannot be made to
    respect limits set afterwards.
    """
    # A segfault inside a generated network is otherwise silent; this turns it
    # into a Python traceback on the worker's stderr, which the trainer shares.
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        pass
    _make_preferred_oom_victim()
    _die_with_parent()
    _limit_address_space()
    # Last, so the guard is already watching while torch and CUDA initialise.
    _start_memory_guard()


def _clear_distributed_env() -> None:
    keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_LOCAL_PROCESS_INDEX",
        "ACCELERATE_NUM_PROCESSES",
    )
    for key in keys:
        os.environ.pop(key, None)


def nneval_worker_main(conn, assigned_gpu=None, assigned_cuda_visible_device=None) -> None:
    _harden_worker_process()
    _clear_distributed_env()
    if assigned_gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        visible_device = (
            str(assigned_cuda_visible_device)
            if assigned_cuda_visible_device is not None
            else str(int(assigned_gpu))
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ab.gpt.util.eval import nneval_worker_pool as NNEvalWorkerPool

    NNEvalWorkerPool._persistent_nneval_worker_entry(conn, assigned_gpu, assigned_cuda_visible_device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NNEval worker bootstrap")
    parser.add_argument("--conn-fd", type=int, required=True)
    parser.add_argument("--assigned-gpu", default="")
    parser.add_argument("--assigned-cuda-visible-device", default="")
    return parser.parse_args()


def _parse_optional_int(raw: str):
    raw = str(raw).strip()
    if raw == "":
        return None
    return int(raw)


def main() -> None:
    args = _parse_args()
    conn = Connection(int(args.conn_fd))
    try:
        nneval_worker_main(
            conn,
            assigned_gpu=_parse_optional_int(args.assigned_gpu),
            assigned_cuda_visible_device=(
                None
                if str(args.assigned_cuda_visible_device).strip() == ""
                else str(args.assigned_cuda_visible_device).strip()
            ),
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
