import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=5, p=0.2),
    transforms.RandomAffine(degrees=17, translate=(0.06, 0.17), scale=(1.03, 1.98), shear=(2.82, 5.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
