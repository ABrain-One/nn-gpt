import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys

# FIX: Import typing helpers globally so exec() can find them in class definitions
from typing import List, Any, Tuple, Dict 

from .logger_utils import MutationLogger

# Global Logger
reward_logger = MutationLogger()

# -----------------------------------------------------------------------
# LOCAL DATA LOADER
# -----------------------------------------------------------------------
def get_cifar10_loader(batch_size=128):
    data_path = os.path.join(os.path.dirname(__file__), "data_v2") 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, 
                                   download=True, transform=transform)
    
    # OPTIMIZATION: Use 10% of data
    # subset_indices = torch.arange(0, len(train_dataset), 10) 
    # train_subset = torch.utils.data.Subset(train_dataset, subset_indices)

# OPTIMIZATION: Use 50% of data (Step=2)
    # Range(0, len, 2) selects indices: 0, 2, 4, 6... (Half the data)
    # subset_indices = torch.arange(0, len(train_dataset), 2) 
    # train_subset = torch.utils.data.Subset(train_dataset, subset_indices)


    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=2)
    return loader

# -----------------------------------------------------------------------
# LOCAL EVALUATOR
# -----------------------------------------------------------------------
def evaluate_fitness(individual, train_conf=None):
    code = individual.code
    
    # 1. Sandbox Compilation
    try:
        # FIX: Use a merged scope. By passing only one dict, it acts as 
        # both globals and locals, fixing the 'List not defined' error.
        # We initialize it with specific globals we want to expose (or just empty)
        local_scope = {}
        
        # Execute the code. 
        # Note: We passed ONLY local_scope. This mimics module-level execution.
        exec(code, local_scope)
        
        if 'Net' not in local_scope:
            raise ValueError("Code must define a 'Net' class")
        
        NetClass = local_scope['Net']
        
    except Exception as e:
        reward_logger.log("COMPILATION_ERROR", code, code, 0.0, str(e))
        return 0.0

    # 2. Training Loop
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Instantiate
        model = NetClass(in_shape=(3, 32, 32), out_shape=(10,), 
                        prm=individual.prm, device=device).to(device)
        
        # Get Data
        train_loader = get_cifar10_loader()
        
        # Setup Optimizer
        if hasattr(model, 'train_setup'):
            optimizer = model.train_setup(individual.prm)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model.train()
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = correct / total
        
        reward_logger.log("EVALUATION_SUCCESS", code, code, accuracy, None)
        return accuracy

    except Exception as e:
        print(f"[Training Error] {e}")
        reward_logger.log("TRAINING_ERROR", code, code, 0.0, str(e))
        return 0.0

# Compatibility Wrapper for run_evolution.py
def evaluate_code_and_reward(code, prm=None, **kwargs):
    """
    Adapts the old API call to the new evaluate_fitness logic.
    """
    if prm is None:
        prm = {'lr': 0.01, 'momentum': 0.9}

    # Create simple mock object to hold the code
    class MockIndividual:
        def __init__(self, c, p):
            self.code = c
            self.prm = p 
            self.chromosome = {'code': c, **p} # Merge prm into chromosome for consistency 

    # Call the new evaluator
    acc = evaluate_fitness(MockIndividual(code, prm))
    
    # Return structure expected by run_evolution.py
    return {
        'val_metric': acc,
        'reward': acc,
        'built_ok': True
    }