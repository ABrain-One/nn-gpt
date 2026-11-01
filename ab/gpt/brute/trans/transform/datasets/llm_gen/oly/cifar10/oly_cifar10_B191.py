import torch<br>
import torchvision.transforms as transforms<br>
&nbsp;<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.8),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomVerticalFlip(p=0.8),<br>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),<br>
&nbsp; &nbsp; &nbsp; transforms.Resize(224),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])<br>
&nbsp; &nbsp; ])