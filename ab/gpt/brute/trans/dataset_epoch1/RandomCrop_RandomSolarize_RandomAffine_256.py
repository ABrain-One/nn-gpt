import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomSolarize(threshold=181, p=0.2),
    transforms.RandomAffine(degrees=26, translate=(0.14, 0.09), scale=(0.82, 1.33), shear=(4.1, 9.45)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
