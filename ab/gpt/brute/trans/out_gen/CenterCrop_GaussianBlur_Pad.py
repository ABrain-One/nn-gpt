import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.68, 1.67)),
    transforms.Pad(padding=0, fill=(180, 36, 73), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
