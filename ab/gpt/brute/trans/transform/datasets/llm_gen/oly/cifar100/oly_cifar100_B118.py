import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(27), transforms.RandomAdjustSharpness(1.47, p=0.25), transforms.ColorJitter(1.16, 0.94, 0.91, 0.08), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(0.73), transforms.RandomEqualize(0.55), transforms.ToTensor(), transforms.Normalize(*norm)])