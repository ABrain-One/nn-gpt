import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.09, 0.01), scale=(0.93, 1.33), shear=(4.94, 6.55)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.54, p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
