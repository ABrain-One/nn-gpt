import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=108, p=0.89),
    transforms.RandomAffine(degrees=19, translate=(0.11, 0.07), scale=(1.14, 1.27), shear=(0.32, 6.34)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
