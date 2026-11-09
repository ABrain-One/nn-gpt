import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(245, 164, 92), padding_mode='reflect'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.86, 1.61)),
    transforms.RandomSolarize(threshold=196, p=0.26),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
