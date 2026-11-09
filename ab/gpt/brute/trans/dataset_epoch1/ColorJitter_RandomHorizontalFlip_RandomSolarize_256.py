import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.01, contrast=1.17, saturation=1.14, hue=0.07),
    transforms.RandomHorizontalFlip(p=0.32),
    transforms.RandomSolarize(threshold=107, p=0.17),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
