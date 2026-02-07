"""
Example: Training Fractal Network with Layerwise Learning Rates
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import the layerwise LR utility
from ab.gpt.util.LayerwiseLR import (
    LayerwiseLRConfig,
    create_optimizer_with_layerwise_lr,
    print_layerwise_lr,
    generate_llm_learning_rates
)

# Import fractal model
from ab.gpt.brute.ga.modular.fractal_seed import Net as FractalNet


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%')

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    base_lr = 0.01

    # Data loading (MNIST for simplicity)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model setup
    in_shape = (1, 28, 28)  # MNIST: 1 channel, 28x28
    out_shape = (10,)       # 10 classes
    prm = {'lr': base_lr, 'dropout': 0.1, 'momentum': 0.9}

    print("\n" + "="*80)
    print("Initializing Fractal Network")
    print("="*80)

    model = FractalNet(in_shape, out_shape, prm, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()

    # ========================================================================
    # Experiment 1: Uniform Learning Rate (Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Uniform Learning Rate (Baseline)")
    print("="*80)

    config_uniform = LayerwiseLRConfig(
        base_lr=base_lr,
        strategy=LayerwiseLRConfig.UNIFORM
    )

    optimizer_uniform = create_optimizer_with_layerwise_lr(
        model, config_uniform,
        optimizer_class=torch.optim.SGD,
        momentum=prm['momentum']
    )

    print_layerwise_lr(optimizer_uniform)

    print("\nTraining with uniform LR...")
    for epoch in range(2):  # Just 2 epochs for demo
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_uniform, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    # ========================================================================
    # Experiment 2: Linear Decay (Early layers smaller LR)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Linear Decay Layerwise LR")
    print("="*80)

    # Re-initialize model for fair comparison
    model = FractalNet(in_shape, out_shape, prm, device)

    config_linear = LayerwiseLRConfig(
        base_lr=base_lr,
        strategy=LayerwiseLRConfig.LINEAR_DECAY,
        decay_factor=0.5  # Last layer will have 0.5x the LR of first layer
    )

    optimizer_linear = create_optimizer_with_layerwise_lr(
        model, config_linear,
        optimizer_class=torch.optim.SGD,
        momentum=prm['momentum']
    )

    print_layerwise_lr(optimizer_linear)

    print("\nTraining with linear decay LR...")
    for epoch in range(2):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_linear, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    # ========================================================================
    # Experiment 3: Discriminative (Different LR for different layer types)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: Discriminative Layerwise LR")
    print("="*80)

    # Re-initialize model
    model = FractalNet(in_shape, out_shape, prm, device)

    config_discriminative = LayerwiseLRConfig(
        base_lr=base_lr,
        strategy=LayerwiseLRConfig.DISCRIMINATIVE,
        layer_lr_map={
            r'entry': 0.5,           # Entry layers: 0.5x
            r'block1': 0.7,          # Block1: 0.7x
            r'trans': 0.8,           # Transition: 0.8x
            r'block2': 0.9,          # Block2: 0.9x
            r'fc': 1.5,              # Final FC: 1.5x (higher LR)
        }
    )

    optimizer_disc = create_optimizer_with_layerwise_lr(
        model, config_discriminative,
        optimizer_class=torch.optim.SGD,
        momentum=prm['momentum']
    )

    print_layerwise_lr(optimizer_disc)

    print("\nTraining with discriminative LR...")
    for epoch in range(2):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_disc, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    # ========================================================================
    # Experiment 4: LLM-Suggested (Heuristic-based for now)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: LLM-Suggested Layerwise LR (Heuristic)")
    print("="*80)

    # Re-initialize model
    model = FractalNet(in_shape, out_shape, prm, device)

    # Generate LLM-suggested rates (currently uses heuristics)
    llm_suggested_rates = generate_llm_learning_rates(
        model,
        base_lr=base_lr,
        model_arch="FractalNet",
        task="classification"
    )

    config_llm = LayerwiseLRConfig(
        base_lr=base_lr,
        strategy=LayerwiseLRConfig.CUSTOM,
        llm_suggested_rates=llm_suggested_rates
    )

    optimizer_llm = create_optimizer_with_layerwise_lr(
        model, config_llm,
        optimizer_class=torch.optim.SGD,
        momentum=prm['momentum']
    )

    print_layerwise_lr(optimizer_llm)

    print("\nTraining with LLM-suggested LR...")
    for epoch in range(2):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_llm, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()
