import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(235, 173, 24), padding_mode='edge'), transforms.GaussianBlur(kernel_size=5, sigma=(0.29, 1.69)), transforms.RandomAffine(degrees=19, translate=(0.06, 0.05), scale=(0.87, 1.67), shear=(2.73, 5.59)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])