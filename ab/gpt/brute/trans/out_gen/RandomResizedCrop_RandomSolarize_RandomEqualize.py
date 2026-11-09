import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.96), ratio=(1.19, 1.47)),
    transforms.RandomSolarize(threshold=252, p=0.15),
    transforms.RandomEqualize(p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
