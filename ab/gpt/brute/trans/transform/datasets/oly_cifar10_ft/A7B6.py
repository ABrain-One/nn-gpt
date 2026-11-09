import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.96), ratio=(1.02, 1.61)),
    transforms.RandomSolarize(threshold=229, p=0.19),
    transforms.RandomEqualize(p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(means=[0.48, 0.41, 0.35], std=[0.96, 0.95, 0.91]),
    ])