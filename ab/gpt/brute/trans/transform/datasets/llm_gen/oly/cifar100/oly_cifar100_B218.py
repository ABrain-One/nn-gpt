import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 2.0)),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomRotation(degrees=(-15, 15)),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),<br/>
&nbsp; &nbsp; &nbsp; transforms.GaussianBlur(kernel_size=3, sigma=1.0),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize(224),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), std=torch.tensor([0.2023, 0.1994, 0.2010]))<br/>
&nbsp; &nbsp; ])