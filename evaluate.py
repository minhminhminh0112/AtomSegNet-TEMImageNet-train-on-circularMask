import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def evaluate(net, dataloader):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device ='cuda', dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device ='cuda',dtype=torch.long)

        # predict the mask
        mask_pred = net(image)

        assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        # compute the Dice score
        dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    net.train() #if need to set the model back to training mode
    return dice_score / max(num_val_batches, 1)

accuracy = evaluate(model, test_loader)
print(accuracy)