import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import time
import csv
import matplotlib.pyplot as plt

# Force matplotlib to not use Xwindows (Cluster safe)
plt.switch_backend('Agg')

# Add current path
sys.path.append(os.getcwd())

def validate():
    print("--- FULL VALIDATION OF BEST INDIVIDUAL ---")
    
    # 1. Load the Best Model
    try:
        from ab.gpt.brute.ga.modular.best_fractal_net import Net
        print("Successfully loaded 'Net' from best_fractal_net.py")
    except ImportError:
        print("Error: best_fractal_net.py not found or invalid.")
        return

    # --- 2. SETUP OUTPUT FOLDER ---
    # Get the directory where this script is located (ab/gpt/brute/ga/modular)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "validationResults")
    
    # Create the folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Define Output Files
    log_file = os.path.join(output_dir, "training_log.csv")
    plot_acc_file = os.path.join(output_dir, "accuracy_plot.png")
    plot_loss_file = os.path.join(output_dir, "loss_plot.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 3. Data Loading
    print("Loading Full CIFAR-10 Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Ensure data is downloaded to root/data_v2 (relative to execution)
    data_dir = os.path.join(os.path.dirname(__file__), "data_v2")
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 4. Initialize Model
    prm = {'drop_path_prob': 0.1, 'dropout': 0.1, 'lr': 0.01, 'momentum': 0.9}
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=device).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    EPOCHS = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Starting Training for {EPOCHS} epochs...")
    
    # Init CSV
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc', 'Time'])

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_test_loss = test_loss / len(testloader)
        test_acc = 100. * correct / total
        epoch_time = time.time() - start

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Log Data
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, train_acc, avg_test_loss, test_acc, epoch_time])

    # 6. Generate Plots
    print("Generating plots...")
    
    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(history['epoch'], history['test_acc'], label='Test Accuracy', marker='o')
    plt.title('FractalNet Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_acc_file)
    print(f"Saved: {plot_acc_file}")
    
    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', linestyle='--')
    plt.plot(history['epoch'], history['test_loss'], label='Test Loss')
    plt.title('FractalNet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_loss_file)
    print(f"Saved: {plot_loss_file}")

    print("Validation Complete.")

if __name__ == "__main__":
    validate()