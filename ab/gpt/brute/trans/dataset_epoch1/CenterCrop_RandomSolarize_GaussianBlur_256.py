import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomSolarize(threshold=173, p=0.36),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.6, 1.91)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
