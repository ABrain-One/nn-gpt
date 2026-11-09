import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.ColorJitter(brightness=1.0, contrast=0.98, saturation=1.01, hue=0.01),
    transforms.RandomSolarize(threshold=58, p=0.37),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
