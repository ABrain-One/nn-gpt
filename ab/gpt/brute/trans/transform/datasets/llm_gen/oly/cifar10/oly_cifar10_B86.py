import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomCrop(30),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomAutocontrast(p=0.5),<br>
&nbsp; &nbsp; &nbsp; transforms.Pad(4, fill=(92, 236, 245), padding_mode='reflect'),<br>
&nbsp; &nbsp; &nbsp; transforms.GaussianBlur(kernel_size=5, sigma=1.5, p=0.5),<br>
&nbsp; &nbsp; &nbsp; transforms.Resize((64, 64)),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])<br>
&nbsp; &nbsp; ])<br>