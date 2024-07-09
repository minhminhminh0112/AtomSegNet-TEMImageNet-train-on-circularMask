
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from data_loading import CustomDataset
from unet_sigmoid import UNet

def train_model(net, dataloader, criterion, optimizer, epochs=10,save_path='unet_model.pth'):
    n_train = len(dataloader)
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in dataloader:
                images, masks = batch['image'], batch['mask']

                # Ensure images and masks are on the correct device
                images = images.to(device='cuda', dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device='cuda', dtype=torch.float32)

                # Forward pass
                outputs = net(images)

                # Compute loss
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(images.shape[0])

        print(f'Epoch {epoch}/{epochs} completed with avg loss: {epoch_loss/len(dataloader):.4f}')

    # Save the model weights
    torch.save(net.state_dict(), save_path)
    print(f'Model saved to {save_path}')
