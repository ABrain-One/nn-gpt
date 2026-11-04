<import>import torch</import>
    <import>import torchvision.transforms as transforms</import>
    <def name="transform">def transform(norm):</def>
    <lines>
        <line>    return transforms.Compose([</line>
        <line>    transforms.RandomVerticalFlip(p=0.73),</line>
        <line>    transforms.RandomAdjustSharpness(sharpness_factor=1.43, p=0.19),</line>
        <line>    transforms.Resize((64,64)),</line>
        <line>    transforms.ToTensor(),</line>
        <line>    transforms.Normalize(*norm)</line>
        <line>)</line>
    </lines>