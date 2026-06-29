"""
Standalone training script for evaluating generated models.
This script is executed in a subprocess to isolate execution.

Usage:
    python train_script.py --model_file <path_to_model_file> --epochs <num_epochs> --output_file <path_to_output>
"""

import json
import sys
import os
import random
import argparse
import tarfile
import traceback
import urllib.request
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Set the seed for reproducibility
SEED = 43

def set_all_seeds(seed=SEED):
    """Set all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Set seeds at module load time
set_all_seeds(SEED)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIR_NAME = "imagenette2-160"
IMAGENETTE_SIZE = 160
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def download_imagenette(data_dir: str):
    """Download and extract ImageNette if not already present."""
    dest = Path(data_dir) / IMAGENETTE_DIR_NAME
    if dest.exists() and (dest / "train").exists():
        return
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    tgz_path = Path(data_dir) / "imagenette2-160.tgz"
    if not tgz_path.exists():
        print(f"Downloading ImageNette to {tgz_path} ...")
        urllib.request.urlretrieve(IMAGENETTE_URL, str(tgz_path))
        print("Download complete.")
    print(f"Extracting {tgz_path} ...")
    with tarfile.open(str(tgz_path), "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")


def load_imagenette(batch_size: int = 128, data_dir: str = './data'):
    """Load ImageNette dataset (160x160, 10 classes)."""
    download_imagenette(data_dir)

    train_dir = os.path.join(data_dir, IMAGENETTE_DIR_NAME, "train")
    val_dir = os.path.join(data_dir, IMAGENETTE_DIR_NAME, "val")

    transform_train = transforms.Compose([
        transforms.Resize((IMAGENETTE_SIZE, IMAGENETTE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMAGENETTE_SIZE, IMAGENETTE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    return train_loader, test_loader


def load_cifar10(batch_size: int = 128, data_dir: str = './data'):
    """Load CIFAR-10 dataset (32x32, 10 classes)."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    return train_loader, test_loader


def load_cifar100(batch_size: int = 128, data_dir: str = './data'):
    """Load CIFAR-100 dataset (32x32, 100 classes)."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """Evaluate the model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate generated model')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--dataset', type=str, default='imagenette',
                        choices=['imagenette', 'cifar10', 'cifar100'], help='Dataset to use')
    args = parser.parse_args()

    result = {
        'success': False,
        'accuracy': 0.0,
        'error': None,
        'train_history': []
    }

    try:
        # Load the model code
        model_file = Path(args.model_file)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {args.model_file}")

        model_code = model_file.read_text()

        # Create a namespace to execute the code
        namespace = {
            'torch': torch,
            'nn': nn,
        }
        # Add common imports that might be used
        exec("import torch.nn.functional as F", namespace)
        exec("from torch import Tensor", namespace)

        # Execute the model code to define the Net class
        exec(model_code, namespace)

        if 'Net' not in namespace:
            raise ValueError("Model code does not define a 'Net' class")

        # Instantiate the model
        Net = namespace['Net']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Net()
        model = model.to(device)
        print(f"Model instantiated successfully on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Reset seeds before training for reproducibility
        set_all_seeds(SEED)
        
        # Load dataset
        if args.dataset == 'cifar100':
            print("Loading CIFAR-100 dataset...")
            train_loader, test_loader = load_cifar100(
                batch_size=args.batch_size,
                data_dir=args.data_dir
            )
        elif args.dataset == 'cifar10':
            print("Loading CIFAR-10 dataset...")
            train_loader, test_loader = load_cifar10(
                batch_size=args.batch_size,
                data_dir=args.data_dir
            )
        else:
            print("Loading ImageNette dataset...")
            train_loader, test_loader = load_imagenette(
                batch_size=args.batch_size,
                data_dir=args.data_dir
            )

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Training loop
        print(f"Starting training for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            scheduler.step()

            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc * 100
            }
            result['train_history'].append(epoch_result)
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%, Test Acc={test_acc*100:.2f}%")

        # Final evaluation
        final_accuracy = evaluate(model, test_loader, device)
        result['success'] = True
        result['accuracy'] = final_accuracy
        print(f"\nFinal Test Accuracy: {final_accuracy*100:.2f}%")

    except Exception as e:
        result['success'] = False
        result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error during training: {result['error']}", file=sys.stderr)

    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_file}")

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
