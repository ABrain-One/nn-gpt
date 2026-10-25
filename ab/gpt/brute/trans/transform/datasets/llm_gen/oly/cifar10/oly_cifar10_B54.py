<td>
        <pre>
<code>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.CenterCrop(29),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.3),
        transforms.RandomPerspective(distortion_scale=0.27, p=0.3),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
</code>
        </pre>
    </td>