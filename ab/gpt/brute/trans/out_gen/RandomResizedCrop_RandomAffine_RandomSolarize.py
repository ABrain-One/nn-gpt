import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.81), ratio=(1.01, 2.09)),
    transforms.RandomAffine(degrees=25, translate=(0.04, 0.11), scale=(1.09, 1.35), shear=(1.37, 9.36)),
    transforms.RandomSolarize(threshold=130, p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
