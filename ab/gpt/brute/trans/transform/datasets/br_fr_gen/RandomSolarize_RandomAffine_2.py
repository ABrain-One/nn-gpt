import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=72, p=0.73),
    transforms.RandomAffine(degrees=10, translate=(0.06, 0.11), scale=(0.8, 1.53), shear=(4.33, 6.0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
