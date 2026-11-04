import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=17, translate=(0.15, 0.06), scale=(0.88, 1.69), shear=(4.33, 7.96)),
    transforms.RandomHorizontalFlip(p=0.47),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
