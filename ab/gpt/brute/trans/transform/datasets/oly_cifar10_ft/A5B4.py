<arg>import torch</arg>
    <arg>import torchvision.transforms as transforms</arg>
    <arg>def transform(norm):</arg>
    <arg>    return transforms.Compose([</arg>
    <arg>        transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.79),</arg>
    <arg>        transforms.RandomEqualize(p=0.75),</arg>
    <arg>        transforms.Resize((64,64)),</arg>
    <arg>        transforms.ToTensor(),</arg>
    <arg>        transforms.Normalize(*norm)</arg>
    <arg>    ])</arg>