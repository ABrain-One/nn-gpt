import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.1, 0.08), scale=(1.09, 1.5), shear=(2.49, 8.91)),
    transforms.RandomSolarize(threshold=25, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
