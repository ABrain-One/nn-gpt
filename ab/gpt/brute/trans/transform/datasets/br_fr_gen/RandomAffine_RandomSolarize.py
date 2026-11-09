import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=22, translate=(0.07, 0.03), scale=(1.17, 1.91), shear=(4.67, 8.5)),
    transforms.RandomSolarize(threshold=154, p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
