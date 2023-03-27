import psutil
from pathlib import Path
import random
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule

num_cpus = len(psutil.Process().cpu_affinity())
ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
QUANTILES = {
    'min_q': {
        'B2': 3.0,
        'B3': 2.0,
        'B4': 0.0
    },
    'max_q': {
        'B2': 88.0,
        'B3': 103.0,
        'B4': 129.0
    }
}


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)



class BaseDataset(Dataset):
    def __init__(self, root, bands=RGB_BANDS):
        super().__init__()
        self.root = Path(root)
        self.bands = bands 
        self._samples = None

    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    distort = transforms.Compose([
        transforms.v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.v2.RandomChoice([
            transforms.v2.RandomGrayscale(p=0.2),
            transforms.v2.RandomAutocontrast(p=0.2),
            transforms.v2.RandomSolarize(p=0.2),
            transforms.v2.RandomPosterize(p=0.2),
            transforms.v2.RandomEqualize(p=0.2),
            transforms.v2.RandomInvert(p=0.2),
        ]),
    ])

    @property
    def samples(self):
        if self._samples is None:
            self._samples = self.get_samples()
        return self._samples

    def get_samples(self):
        # return [path for path in self.root.glob('*') if path.is_dir()]
        return [path for path in self.root.glob('[0-9]*') if path.is_dir()]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        root = self.samples[index]
        sorted_paths = sorted([path for path in root.glob('*') if path.is_dir()], reverse=True)
        t1, t2 = [read_image(path, self.bands, QUANTILES) for path in np.random.choice(sorted_paths, 2)]
        # Apply the same random crop to query image and image to be reconstructed by the decoder
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        img1 = self.preprocess(t1)
        img1 = self.distort(img1)
        torch.manual_seed(seed)
        img2 = self.preprocess(t2)

        return img1, img2

def normalize(img, min_q, max_q):
    img = (img - min_q) / (max_q - min_q)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def read_image(path, bands, quantiles=None):
    channels = []
    for b in bands:
        ch = rasterio.open(path / f'{b}.tif').read(1)
        if quantiles is not None:
            ch = normalize(ch, min_q=quantiles['min_q'][b], max_q=quantiles['max_q'][b])
        channels.append(ch)
    img = np.dstack(channels)
    img = Image.fromarray(img)
    return img

class CDDataModule(LightningDataModule):
    def __init__(self, data_dir, bands=RGB_BANDS, batch_size=4, num_workers=num_cpus, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset = None
        self.train_transforms = None

    def setup(self, stage=None):
        self.train_dataset = self.get_dataset(root=self.data_dir, bands=self.bands)

    @staticmethod
    def get_dataset(root, bands):
        return BaseDataset(root, bands)

    def train_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return InfiniteDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )





