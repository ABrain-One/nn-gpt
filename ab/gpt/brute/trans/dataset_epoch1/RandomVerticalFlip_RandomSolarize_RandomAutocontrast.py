import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.51),
    transforms.RandomSolarize(threshold=92, p=0.86),
    transforms.RandomAutocontrast(p=0.57),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
