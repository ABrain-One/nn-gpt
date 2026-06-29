import gc
import os
from pathlib import Path


def _read_process_rss_gib() -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        return None
    return None


def torchvision_prewarm_main(conn, backbone_names: list[str]) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        import torchvision

        checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        conn.send(
            {
                "cmd": "prewarm_ready",
                "pid": os.getpid(),
                "torch_home": os.environ.get("TORCH_HOME", ""),
                "checkpoints_dir": str(checkpoints_dir),
                "rss_gib": float(_read_process_rss_gib() or 0.0),
            }
        )

        completed = 0
        failed = 0
        for backbone_name in backbone_names:
            checkpoint_before = sorted(p.name for p in checkpoints_dir.glob("*"))
            cache_hit = False
            try:
                if hasattr(torchvision.models, "get_model"):
                    model = torchvision.models.get_model(backbone_name, weights="DEFAULT")
                else:
                    model = torchvision.models.__dict__[backbone_name](pretrained=True)
                del model
                gc.collect()
                checkpoint_after = sorted(p.name for p in checkpoints_dir.glob("*"))
                cache_hit = checkpoint_before == checkpoint_after
                completed += 1
                conn.send(
                    {
                        "cmd": "prewarm_progress",
                        "backbone": backbone_name,
                        "cache_hit": cache_hit,
                        "completed": completed,
                        "failed": failed,
                        "checkpoints_dir": str(checkpoints_dir),
                        "rss_gib": float(_read_process_rss_gib() or 0.0),
                    }
                )
            except Exception as exc:
                failed += 1
                conn.send(
                    {
                        "cmd": "prewarm_error",
                        "backbone": backbone_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed": completed,
                        "failed": failed,
                        "checkpoints_dir": str(checkpoints_dir),
                        "rss_gib": float(_read_process_rss_gib() or 0.0),
                    }
                )
                raise

        conn.send(
            {
                "cmd": "prewarm_done",
                "completed": completed,
                "failed": failed,
                "checkpoints_dir": str(checkpoints_dir),
                "rss_gib": float(_read_process_rss_gib() or 0.0),
            }
        )
    except Exception as exc:
        try:
            conn.send(
                {
                    "cmd": "prewarm_fatal",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass
