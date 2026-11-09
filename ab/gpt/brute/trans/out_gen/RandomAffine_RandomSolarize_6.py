import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=2, translate=(0.17, 0.02), scale=(0.99, 1.96), shear=(4.86, 5.74)),
    transforms.RandomSolarize(threshold=113, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
