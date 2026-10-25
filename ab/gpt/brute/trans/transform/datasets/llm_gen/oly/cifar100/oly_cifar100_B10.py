import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.Resize(256), T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.99, 1.01)), T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), std=torch.tensor([0.2023, 0.1994, 0.201]))])