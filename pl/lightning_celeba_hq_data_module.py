from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

from datasets.flat_image_folder import FlatImageFolder


class LightningCelebaHqDataModule(LightningDataModule):
    celeba_hq_train: Dataset
    celeba_hq_val: Dataset
    celeba_hq_test: Dataset

    def __init__(self, image_size: int, image_channels: int, data_dir: str, batch_size):
        super().__init__()
        assert image_channels == 1 or image_channels == 3
        self.needs_resize = False
        self.available_sizes = [128, 256, 512, 1024]

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels = image_channels

        if self.image_size not in self.available_sizes:
            self.needs_resize = True
            self.nearest_size = min(filter(lambda s: s > self.image_size, self.available_sizes))
            self.data_dir += "/celebAHQ/data" + str(self.nearest_size) + "x" + str(self.nearest_size)
        else:
            self.nearest_size = image_size
            self.data_dir += "/celebAHQ/data" + str(self.image_size) + "x" + str(self.image_size)

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (self.image_channels, self.image_size, self.image_size)

    def setup(self, stage: Optional[str] = None):
        transform_ops = [
            transforms.Resize(self.image_size) if self.needs_resize else None,
            transforms.Grayscale() if self.image_channels == 1 else None,
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if self.image_channels == 1
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        transform = transforms.Compose([op for op in transform_ops if op is not None])
        celeba_hq_full = FlatImageFolder(self.data_dir, transform=transform)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.celeba_hq_train, self.celeba_hq_val = random_split(celeba_hq_full, [27000, 3000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.celeba_hq_test = celeba_hq_full

    def prepare_data(self):
        # Download data etc.
        pass

    def train_dataloader(self):
        return DataLoader(self.celeba_hq_train, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.celeba_hq_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.celeba_hq_test, batch_size=self.batch_size)
