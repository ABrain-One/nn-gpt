"""
Evaluator module.
Trains and evaluates generated models using subprocess isolation.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    success: bool
    accuracy: Optional[float]
    error: Optional[str]
    train_history: list
    
    def get_feedback(self) -> str:
        """Get feedback message for prompt improver."""
        if self.success:
            return f"Training successful. Test accuracy: {self.accuracy * 100:.2f}%"
        else:
            return f"Training failed with error: {self.error}"


class Evaluator:
    """Evaluates generated model code by training on a chosen dataset."""
    
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        timeout: int = 1800,  # 30 minutes default timeout
        data_dir: str = './data',
        dataset: str = 'imagenette'
    ):
        """
        Initialize the evaluator.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            timeout: Maximum time in seconds for training
            data_dir: Directory for dataset data
            dataset: Dataset to use ('imagenette', 'cifar10', or 'cifar100')
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.timeout = timeout
        self.data_dir = data_dir
        self.dataset = dataset
        
        # Get the directory of this file for finding train_script.py
        self.script_dir = Path(__file__).parent
        self.train_script = self.script_dir / "train_script.py"
    
    def train_and_evaluate(self, code: str) -> EvaluationResult:
        """
        Train the model and evaluate its accuracy.
        
        Args:
            code: Python code defining the Net class
            
        Returns:
            EvaluationResult with accuracy or error information
        """
        # Create temporary files for model code and results
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as model_file:
            model_file.write(code)
            model_path = model_file.name
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as result_file:
            result_path = result_file.name
        
        try:
            # Run training in subprocess
            cmd = [
                sys.executable,
                str(self.train_script),
                '--model_file', model_path,
                '--epochs', str(self.epochs),
                '--output_file', result_path,
                '--batch_size', str(self.batch_size),
                '--lr', str(self.learning_rate),
                '--data_dir', self.data_dir,
                '--dataset', self.dataset
            ]
            
            # Set environment variables for reproducibility
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = '43'
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for CUDA deterministic ops
            
            print(f"Running training subprocess...")
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )
            
            # Print subprocess output
            if process.stdout:
                print("Training output:")
                print(process.stdout)
            
            if process.stderr:
                print("Training errors:", file=sys.stderr)
                print(process.stderr, file=sys.stderr)
            
            # Load results
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                
                return EvaluationResult(
                    success=result_data.get('success', False),
                    accuracy=result_data.get('accuracy'),
                    error=result_data.get('error'),
                    train_history=result_data.get('train_history', [])
                )
            else:
                return EvaluationResult(
                    success=False,
                    accuracy=None,
                    error=f"Result file not created. Subprocess exit code: {process.returncode}",
                    train_history=[]
                )
                
        except subprocess.TimeoutExpired:
            return EvaluationResult(
                success=False,
                accuracy=None,
                error=f"Training timed out after {self.timeout} seconds",
                train_history=[]
            )
        except Exception as e:
            return EvaluationResult(
                success=False,
                accuracy=None,
                error=f"Evaluation error: {type(e).__name__}: {str(e)}",
                train_history=[]
            )
        finally:
            # Clean up temporary files
            try:
                os.unlink(model_path)
            except:
                pass
            try:
                os.unlink(result_path)
            except:
                pass
    
    def quick_validate(self, code: str) -> tuple[bool, str]:
        """
        Quickly validate code without full training.
        Checks if the model can be instantiated and performs a forward pass.
        
        Args:
            code: Python code defining the Net class
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            import torch
            import torch.nn as nn
            
            # Create namespace and execute code
            namespace = {
                'torch': torch,
                'nn': nn,
            }
            exec("import torch.nn.functional as F", namespace)
            exec("from torch import Tensor", namespace)
            exec(code, namespace)
            
            if 'Net' not in namespace:
                return False, "Code does not define a 'Net' class"
            
            # Try to instantiate
            Net = namespace['Net']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                model = Net(parameters={})
            except TypeError:
                model = Net()
            
            model = model.to(device)
            
            # Test forward pass with dummy input matching dataset shape
            if getattr(self, 'dataset', 'imagenette') in ['cifar10', 'cifar100']:
                dummy_input = torch.randn(2, 3, 32, 32).to(device)
            else:
                dummy_input = torch.randn(2, 3, 160, 160).to(device)
            output = model(dummy_input)
            
            # Check output shape (should be batch_size x num_classes)
            expected_classes = 100 if getattr(self, 'dataset', 'imagenette') == 'cifar100' else 10
            if output.shape[0] != 2:
                return False, f"Output batch size mismatch: expected 2, got {output.shape[0]}"
            if output.shape[1] != expected_classes:
                return False, f"Output classes mismatch: expected {expected_classes}, got {output.shape[1]}"
            
            return True, f"Model validated successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}"
            
        except Exception as e:
            return False, f"Validation error: {type(e).__name__}: {str(e)}"


if __name__ == "__main__":
    # Test with a simple model (ImageNette: 3x160x160 input, 10 classes)
    test_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, parameters: dict = None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""
    
    evaluator = Evaluator(epochs=1)  # Quick test with 1 epoch
    
    # Quick validation
    is_valid, msg = evaluator.quick_validate(test_code)
    print(f"Quick validation: {is_valid}, {msg}")
    
    # Full evaluation (uncomment to test)
    # result = evaluator.train_and_evaluate(test_code)
    # print(f"Evaluation result: {result}")
