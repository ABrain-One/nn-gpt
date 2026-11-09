<td>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(191, 254, 0), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.88, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
</td>