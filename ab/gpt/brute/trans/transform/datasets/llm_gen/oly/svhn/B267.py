import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3333)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.RandomSolarize(threshold=50, p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.ToTensor(), transforms.Normalize(mean=torch.tensor([0.4366]), std=torch.tensor([0.2675]))])