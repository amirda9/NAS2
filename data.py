import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from pathlib import Path

class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        else:
            self.train = torch.utils.data.DataLoader(
                Dt(root='./data', train=True, transform=transform, download=True),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=2, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root='./data', train=False, transform=transform),
                batch_size=args.dis_bs, shuffle=False,
                num_workers=2, pin_memory=True)

            self.test = self.valid


class ImagenetteDataset(object):
    def __init__(self, patch_size=32, validation=False, should_normalize=True):
        if not os.path.isfile('imagenette2-320.tgz'):
            # download imagenette dataset
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
            os.system('wget ' + url)
            os.system('tar -xvf imagenette2-320.tgz')
            # !wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
            # !tar -xvf imagenette2-320.tgz

        self.folder = Path('imagenette2-320/train') if not validation else Path('imagenette2-320/val')
        self.classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']

        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)
        
        self.patch_size = patch_size
        self.validation = validation
        
        self.random_resize = torchvision.transforms.RandomResizedCrop(patch_size)
        self.center_resize = torchvision.transforms.CenterCrop(patch_size)
        self.should_normalize = should_normalize
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        image_fname = self.images[index]
        image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)
        
        if not self.validation: image = self.random_resize(image)
        else: image = self.center_resize(image)
            
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, 32, 32)
        if self.should_normalize: image = self.normalize(image)
        
        return image, label
    # get loader
    def get_loader(self, batch_size, shuffle=True, num_workers=4):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __len__(self):
        return len(self.images)


