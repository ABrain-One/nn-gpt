
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import shutil
import torch
import gc

def export_llm_to_onnx(
    pt_model, 
    save_dir: Path, 
    tokenizer: PreTrainedTokenizer,
    opset: int = 17,
    cleanup_temp: bool = True
):
    """
    Export a PyTorch CausalLM model to ONNX using Optimum.
    
    Args:
        pt_model: PyTorch model (should already be unquantized FP16/BF16)
        save_dir: Directory to save ONNX model
        tokenizer: Tokenizer for the model
        opset: ONNX opset version (default 17)
        cleanup_temp: Whether to delete temporary files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Save model directly (should already be clean FP16)
    tmp_dir = save_dir / "tmp_pytorch_checkpoint"
    tmp_dir.mkdir(exist_ok=True)
    
    print(f"  [1/3] Saving model to temporary checkpoint: {tmp_dir}")
    pt_model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    
    # Free memory
    del pt_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step 2: Convert to ONNX using Optimum (FIXED - removed opset and use_io_binding)
    print(f"  [2/3] Converting to ONNX... This may take 1-2 minutes")
    try:
        # The export=True triggers conversion, no need to pass opset here
        ort_model = ORTModelForCausalLM.from_pretrained(
            tmp_dir,
            export=True,  # This triggers the ONNX export
            provider="CUDAExecutionProvider",
        )
        
        # Step 3: Save ONNX model
        print(f"  [3/3] Saving ONNX model to {save_dir}")
        ort_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        print(f"  ✓ ONNX export successful: {save_dir / 'model.onnx'}")
        
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if cleanup_temp and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  Cleaned up temporary files")
        
        if 'ort_model' in locals():
            del ort_model
        
        torch.cuda.empty_cache()
        gc.collect()
