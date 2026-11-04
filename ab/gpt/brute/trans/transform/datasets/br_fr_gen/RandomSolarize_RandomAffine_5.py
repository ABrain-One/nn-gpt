import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=126, p=0.39),
    transforms.RandomAffine(degrees=2, translate=(0.11, 0.17), scale=(1.09, 1.77), shear=(3.3, 6.5)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
