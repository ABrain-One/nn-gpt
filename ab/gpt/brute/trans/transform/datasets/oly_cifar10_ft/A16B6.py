<td><code>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(135, 75, 209), padding_mode='constant'),
    transforms.RandomRotation(degrees=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.48, std=0.29, max_pixel_value=255, inplace=False),
    transforms.Resize((64,64))
])
</code></td>