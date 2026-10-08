import argparse
import os
from multiprocessing.connection import Connection


def _clear_distributed_env() -> None:
    # Reward workers must behave like isolated single-process evaluators.
    # Inheriting torchrun/accelerate env can make nested formal eval pick up
    # the outer distributed topology and bind work to the wrong GPU.
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


def reward_worker_main(conn, assigned_gpu=None, assigned_cuda_visible_device=None) -> None:
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
    from ab.gpt.util import Reward as RewardUtil

    RewardUtil._persistent_eval_worker_entry(conn, assigned_gpu, assigned_cuda_visible_device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward worker bootstrap")
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
        reward_worker_main(
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
