from torchvision import transforms
from torch.utils.data import random_split
from data_loading import CustomDataset
from torch.utils.data import DataLoader
from unet_sigmoid import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from train import train_model
from evaluate import evaluate

'''
#clone dataset
!git init
!git remote add -f origin https://github.com/xinhuolin/TEM-ImageNet-v1.3.git
!git config core.sparseCheckout true
!echo "circularMask" >> .git/info/sparse-checkout
!echo "image" >> .git/info/sparse-checkout
!git pull origin master
'''

# Add correct path to the image directory and mask directory
image_dir = 'image' 
mask_dir = 'circularMask'   

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomDataset(img_dir=image_dir, mask_dir=mask_dir, transform=transform)
train_set, val_set = random_split(dataset, [0.75, 0.25], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, shuffle=False)

# Initialize model, criterion, and optimizer
model = UNet(colordim=1).cuda()
criterion = nn.MSELoss() #nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model and save weights
train_model(model, epochs=2, save_path='unet_MSE_loss.pth')

accuracy = evaluate(model, val_loader)
print(accuracy)