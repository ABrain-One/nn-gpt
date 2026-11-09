<td>
<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(padding=4, fill=(172, 150, 24), padding_mode='constant'),
        transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-5,5)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(**norm)
    ])
</pre>
</td>