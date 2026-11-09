import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(33, 224, 221), padding_mode='reflect'),
    transforms.RandomSolarize(threshold=102, p=0.36),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.69, 1.8)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
