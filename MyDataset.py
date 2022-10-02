import os

import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MyDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, classes=None):
        samples = []
        for fn in os.listdir(root):
            full_path = root + "/" + fn
            label = fn.split('_')[0]
            item = full_path, label
            #print(full_path)
            samples.append(item)
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        elif self.classes is not None:
            target = self.__default_target__(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
    
    def __default_target__(self, target):
        for i in range(len(self.classes)):
            if target == self.classes[i]:
                return i
        return -1

def main():
    import torch

    # Transform PIL to Tensor
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    # Data
    #evaluationset = torchvision.datasets.ImageFolder(root='./LCC_FASD/development', transform=transform)
    #evaluationset = ImageFolderWithPaths(root='./LCC_FASD/evaluation', transform=transform)
    evaluationset = MyDataset(root='./LCC_FASD/evaluation', transform=transform)
    evaluationLoader = torch.utils.data.DataLoader(evaluationset, batch_size=8, shuffle=False, num_workers=2)

    for data in evaluationLoader:
        #print(data)
        inputs, labels = data
        print(inputs.size())
        print(labels)
        exit()

if __name__ == '__main__':
    main()