import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.15), transforms.Pad(padding=0, fill=(242, 129, 127), padding_mode='edge'), transforms.RandomGrayscale(p=0.86), transforms.RandomResizedCrop(size=32, scale=(0.78, 0.86), ratio=(1.01, 1.88)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])