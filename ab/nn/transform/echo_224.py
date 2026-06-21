import torchvision.transforms as transforms


def transform(_):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
