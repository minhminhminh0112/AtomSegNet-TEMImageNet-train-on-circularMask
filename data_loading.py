import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        self.transform = transform

        # Debugging: Print the first few paths to verify
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
        print(f"Sample image path: {self.image_paths[:2]}")
        print(f"Sample mask path: {self.mask_paths[:2]}")

        # Ensure the lengths match
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between number of images and masks"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}