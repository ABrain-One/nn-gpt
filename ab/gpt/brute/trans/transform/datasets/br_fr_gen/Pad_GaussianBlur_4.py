import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(238, 96, 239), padding_mode='symmetric'),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.33, 1.34)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
