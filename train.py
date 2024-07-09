
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from data_loading import CustomDataset
from unet_sigmoid import UNet


image_dir = 'image'  # Add correct path to the image directory
mask_dir = 'circularMask'    # Add correct path to the mask directory

def train_model(model, epochs=10, save_path='unet_model.pth'):
    n_train = len(train_set)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, masks = batch['image'], batch['mask']

                # Ensure images and masks are on the correct device
                images = images.to(device='cuda', dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device='cuda', dtype=torch.float32)

                # Forward pass
                outputs = model(images)

                # Compute loss
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(images.shape[0])

        print(f'Epoch {epoch}/{epochs} completed with avg loss: {epoch_loss/len(train_loader):.4f}')

    # Save the model weights
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomDataset(img_dir=image_dir, mask_dir=mask_dir, transform)
train_set, val_set = random_split(dataset, [0.75, 0.25], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, shuffle=False)

# Initialize model, criterion, and optimizer
model = UNet(colordim=1).cuda()
criterion = nn.MSELoss() #nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model and save weights
train_model(model, epochs=2, save_path='unet_MSE_loss.pth')
