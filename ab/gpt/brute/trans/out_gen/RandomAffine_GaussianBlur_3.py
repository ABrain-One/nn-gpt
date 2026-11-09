import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=1, translate=(0.09, 0.03), scale=(0.91, 1.42), shear=(1.04, 6.83)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.67, 1.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
