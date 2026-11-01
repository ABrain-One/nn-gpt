<block>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=3, fill=(34, 198, 82), padding_mode='symmetric'),
        transforms.RandomAdjustSharpness(sharpness_factor=1.33, p=0.63),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])
</block>