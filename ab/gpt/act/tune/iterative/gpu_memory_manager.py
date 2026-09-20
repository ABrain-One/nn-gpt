#!/usr/bin/env python3
"""
GPU Memory Management Utilities

Provides functions to monitor, clear, and manage GPU memory
to prevent OOM errors during fine-tuning.
"""

import gc
import logging
import subprocess
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get GPU memory information.
    
    Returns:
        (total_gb, used_gb, free_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        free = total - reserved
        
        return total, reserved, free
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return 0.0, 0.0, 0.0


def clear_gpu_cache():
    """Clear PyTorch GPU cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")


def kill_gpu_processes(exclude_pids: Optional[list] = None):
    """
    Kill processes using GPU (except excluded PIDs).
    
    Args:
        exclude_pids: List of PIDs to exclude from killing
    """
    try:
        # Use nvidia-smi to find processes using GPU
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return
        
        pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
        exclude_pids = exclude_pids or []
        
        killed = 0
        for pid in pids:
            if pid and pid not in exclude_pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False, capture_output=True)
                    killed += 1
                except Exception:
                    pass
        
        if killed > 0:
            logger.warning(f"Killed {killed} GPU processes to free memory")
            clear_gpu_cache()
    except Exception as e:
        logger.warning(f"Failed to kill GPU processes: {e}")


def check_gpu_memory(min_free_gb: float = 5.0) -> Tuple[bool, str]:
    """
    Check if GPU has sufficient free memory.
    
    Args:
        min_free_gb: Minimum free memory required in GB
    
    Returns:
        (has_sufficient, message)
    """
    total, used, free = get_gpu_memory_info()
    
    if total == 0:
        return False, "GPU not available"
    
    if free < min_free_gb:
        return False, f"Insufficient GPU memory: {free:.2f}GB free, need {min_free_gb}GB"
    
    return True, f"GPU memory OK: {free:.2f}GB free ({used:.2f}GB/{total:.2f}GB used)"


def ensure_gpu_memory(min_free_gb: float = 5.0, aggressive: bool = False) -> bool:
    """
    Ensure GPU has sufficient memory, clearing cache if needed.
    
    Args:
        min_free_gb: Minimum free memory required
        aggressive: If True, kill other GPU processes
    
    Returns:
        True if sufficient memory available
    """
    # Clear cache first
    clear_gpu_cache()
    
    # Check memory
    has_sufficient, msg = check_gpu_memory(min_free_gb)
    
    if has_sufficient:
        logger.info(msg)
        return True
    
    logger.warning(f"GPU memory check failed: {msg}")
    
    if aggressive:
        logger.info("Attempting aggressive memory cleanup...")
        kill_gpu_processes()
        clear_gpu_cache()
        
        # Check again
        has_sufficient, msg = check_gpu_memory(min_free_gb)
        if has_sufficient:
            logger.info(f"Memory freed: {msg}")
            return True
    
    return False


def get_recommended_batch_size(current_batch_size: int, gpu_memory_gb: float) -> int:
    """
    Get recommended batch size based on GPU memory.
    
    Args:
        current_batch_size: Current batch size
        gpu_memory_gb: Available GPU memory in GB
    
    Returns:
        Recommended batch size
    """
    # Rough estimate: 7B model needs ~15GB for batch_size=1 with gradient checkpointing
    # Scale linearly
    if gpu_memory_gb < 10:
        return 1
    elif gpu_memory_gb < 15:
        return 1
    elif gpu_memory_gb < 20:
        return 1  # Keep at 1 for safety
    else:
        return min(current_batch_size, 2)  # Max 2 for 7B model


if __name__ == "__main__":
    # Test GPU memory functions
    logging.basicConfig(level=logging.INFO)
    
    total, used, free = get_gpu_memory_info()
    print(f"GPU Memory: {free:.2f}GB free / {total:.2f}GB total")
    
    has_sufficient, msg = check_gpu_memory(5.0)
    print(f"Memory check: {msg}")



