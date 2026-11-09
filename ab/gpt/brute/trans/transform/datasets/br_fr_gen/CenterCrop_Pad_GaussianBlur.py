import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.Pad(padding=1, fill=(22, 220, 159), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.54, 1.98)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
