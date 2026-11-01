import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.RandomAffine(degrees=4, translate=(0.09, 0.09), scale=(1.11, 1.68), shear=(0.12, 6.5)),
    transforms.RandomEqualize(p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
