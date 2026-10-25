tag, is:

< tr>
    import torch<br/>
    import torchvision.transforms as transforms<br/>
    <br/>
    def transform(norm):<br/>
        return transforms.Compose([<br/>
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.75, 1.3)),<br/>
            transforms.RandomGrayscale(p=0.25),<br/>
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
            transforms.RandomHorizontalFlip(p=0.5),<br/>
            transforms.RandomVerticalFlip(p=0.5),<br/>
            transforms.ToTensor(),<br/>
            transforms.Normalize(*norm)<br/>
        ])