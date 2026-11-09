import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(230, 178, 6), padding_mode='constant'),
    transforms.RandomGrayscale(p=0.7),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.52, 1.24)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
