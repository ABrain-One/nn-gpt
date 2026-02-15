"""Fine-tuning script for multiple models on CIFAR-10"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time


class CIFARTrainer:
    def __init__(self, model_name='ResNet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Dataset setup
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        
    def _setup_data(self):
        """Load CIFAR-10 dataset"""
        print(f"Loading CIFAR-10 dataset...")
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=2
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=2
        )
        
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    def _load_model(self):
        """Load model based on model_name"""
        print(f"Loading {self.model_name}...")
        
        model_dict = {
            'ResNet18': models.resnet18(weights=None),
            'ResNet34': models.resnet34(weights=None),
            'ResNet50': models.resnet50(weights=None),
            'ResNet101': models.resnet101(weights=None),
            'VGG16': models.vgg16(weights=None),
            'VGG19': models.vgg19(weights=None),
            'MobileNetV2': models.mobilenet_v2(weights=None),
            'MobileNetV3_Small': models.mobilenet_v3_small(weights=None),
            'MobileNetV3_Large': models.mobilenet_v3_large(weights=None),
            'DenseNet121': models.densenet121(weights=None),
            'DenseNet169': models.densenet169(weights=None),
            'EfficientNet_B0': models.efficientnet_b0(weights=None),
            'EfficientNet_B1': models.efficientnet_b1(weights=None),
            'SqueezeNet': models.squeezenet1_0(weights=None),
            'InceptionV3': models.inception_v3(weights=None),
            'GoogleNet': models.googlenet(weights=None),
            'ShuffleNetV2': models.shufflenet_v2_x1_0(weights=None),
            'AlexNet': models.alexnet(weights=None),
        }
        
        if self.model_name not in model_dict:
            raise ValueError(f"Model {self.model_name} not supported. Available: {list(model_dict.keys())}")
        
        self.model = model_dict[self.model_name]
        
        # Adapt final layer to CIFAR-10 (10 classes)
        self._adapt_model_head()
        self.model = self.model.to(self.device)
        print(f"✓ Model loaded: {self.model_name}")
    
    def _adapt_model_head(self):
        """Adapt model's classification head to CIFAR-10 (10 classes)"""
        # ResNet variants
        if 'ResNet' in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        # VGG variants
        elif 'VGG' in self.model_name:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 10)
        # MobileNet variants
        elif 'MobileNet' in self.model_name:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 10)
        # DenseNet variants
        elif 'DenseNet' in self.model_name:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 10)
        # EfficientNet variants
        elif 'EfficientNet' in self.model_name:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 10)
        # SqueezeNet
        elif 'SqueezeNet' in self.model_name:
            self.model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))
        # InceptionV3
        elif 'InceptionV3' in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
            self.model.aux_logits = False
        # GoogleNet
        elif 'GoogleNet' in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        # ShuffleNet
        elif 'ShuffleNet' in self.model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        # AlexNet
        elif 'AlexNet' in self.model_name:
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 10)
        
    def train_epoch(self, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{'~' + str(len(self.train_loader))}: "
                      f"Loss={loss.item():.4f}, Acc={100. * correct / total:.2f}%")
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, epochs=10, lr=0.01, scheduler_type='step'):
        """Complete training loop"""
        self._load_model()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Setup scheduler
        scheduler = None
        if scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'exponential':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        print(f"\nTraining {self.model_name} for {epochs} epochs")
        print(f"Learning Rate: {lr}, Scheduler: {scheduler_type}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        results = {
            'model': self.model_name,
            'scheduler': scheduler_type,
            'epochs': epochs,
            'lr': lr,
            'train_losses': [],
            'train_accs': [],
            'test_accs': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(optimizer, scheduler)
            test_acc = self.test()
            
            results['train_losses'].append(train_loss)
            results['train_accs'].append(train_acc)
            results['test_accs'].append(test_acc)
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Acc:  {test_acc:.2f}%")
            print()
        
        elapsed_time = time.time() - start_time
        results['training_time'] = elapsed_time
        
        print(f"Training completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best Test Accuracy: {max(results['test_accs']):.2f}%")
        
        return results


def train_multiple_models(model_names=['ResNet18', 'ResNet34', 'MobileNetV2'], 
                         num_epochs=1, learning_rate=0.001, 
                         scheduler_types=['exponential']):
    """Train multiple models on CIFAR-10"""
    results_all = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}\n")
        
        trainer = CIFARTrainer(model_name=model_name)
        results = trainer.train(num_epochs=num_epochs, learning_rate=learning_rate, 
                               scheduler_types=scheduler_types)
        results_all[model_name] = results
        
        print(f"{'='*60}\n")
    
    return results_all


def main():
    """Main entry point"""
    # Train multiple models
    models_to_train = [
        'ResNet18',
        'ResNet34', 
        'ResNet50',
        'MobileNetV2',
        'MobileNetV3_Small',
        'DenseNet121',
        'VGG16',
        'EfficientNet_B0',
    ]
    
    print("\n" + "="*60)
    print("CIFAR-10 Multi-Model Fine-tuning")
    print("="*60 + "\n")
    
    results = train_multiple_models(
        model_names=models_to_train,
        num_epochs=1,
        learning_rate=0.001,
        scheduler_types=['exponential']
    )
    
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60 + "\n")
    
    for model_name, result in results.items():
        best_acc = max(result['test_accs'])
        train_time = result['training_time']
        print(f"{model_name:20} | Accuracy: {best_acc:6.2f}% | Time: {train_time/60:6.2f} min")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()


def main():
    """Main fine-tuning execution"""
    print("\n" + "=" * 70)
    print("ResNet18 Fine-Tuning on CIFAR-10")
    print("=" * 70)
    
    # Initialize database
    print("\nInitializing database...")
    init_database()
    print("✓ Database initialized")
    
    # Configuration
    model_name = ACTIVE_MODEL  # ResNet18
    epochs = 10
    lr = 0.01
    schedulers = ['step', 'cosine', 'exponential']
    
    # Train with different schedulers
    all_results = []
    
    for scheduler_type in schedulers:
        print(f"\n{'=' * 70}")
        print(f"Training with {scheduler_type.upper()} scheduler")
        print(f"{'=' * 70}")
        
        trainer = CIFARTrainer(model_name=model_name)
        results = trainer.train(epochs=epochs, lr=lr, scheduler_type=scheduler_type)
        
        # Save results to database
        print(f"\nSaving results to database...")
        save_scheduler_result(
            model_name=model_name,
            scheduler_type=scheduler_type,
            epoch=epochs,
            learning_rate=lr,
            best_accuracy=max(results['test_accs']),
            final_accuracy=results['test_accs'][-1],
            training_time=results['training_time']
        )
        print("✓ Results saved")
        
        all_results.append(results)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for results in all_results:
        best_acc = max(results['test_accs'])
        final_acc = results['test_accs'][-1]
        print(f"\n{results['scheduler'].upper()} Scheduler:")
        print(f"  Best Accuracy:  {best_acc:.2f}%")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Training Time:  {results['training_time'] / 60:.2f} minutes")
    
    print("\n✓ Fine-tuning completed successfully!")


if __name__ == '__main__':
    main()
