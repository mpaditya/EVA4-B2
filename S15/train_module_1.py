# -*- coding: utf-8 -*-

import torch, torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
from loss_module import loss_mask_function, loss_depth_function

def save_plot(tensors, name):
    grid_tensor = torchvision.utils.make_grid(tensors)
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize=(20,20))
    plt.imshow(grid_image)
    plt.savefig(name, bbox_inches = 'tight')
    plt.close()
    
def train(model, device, train_loader, optimizer, epoch,
          l1_decay, l2_decay, train_losses, scheduler=None):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  avg_loss = 0
  for batch_idx, data in enumerate(pbar):
    # get samples
    bg = data["f1"]
    fg_bg = data["f2"]
    fg_bg_mask = data["f3"].to(device)
    fg_bg_depth = data["f4"].to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    inp = torch.cat([bg,fg_bg], dim=1)
    inp = inp.to(device)
    mask_pred, depth_pred = model(inp)

    # Calculate loss
    loss_mask, mstr = loss_mask_function(output=mask_pred, target=fg_bg_mask,
                          w_depth=1.0, w_ssim=1.0, w_edge=1.0)
    loss_depth, dstr = loss_depth_function(output=depth_pred, target=fg_bg_depth,
                          w_depth=1.0, w_ssim=1.0, w_edge=1.0)
    loss = loss_mask + loss_depth
    if l1_decay > 0:
      l1_loss = 0
      for param in model.parameters():
        l1_loss += torch.norm(param,1)
      loss += l1_decay * l1_loss
    if l2_decay > 0:
      l2_loss = 0
      for param in model.parameters():
        l2_loss += torch.norm(param,2)
      loss += l2_decay * l2_loss

    # Backpropagation
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()


    if (batch_idx+1)%500 == 0:
      torch.save(model.state_dict(), f"/content/models/e{epoch:03d}_b{batch_idx:05d}_l{loss.item():0.5f}.pth")

    # Update pbar-tqdm
    # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    # correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(inp)
    avg_loss += loss.item()

    pbar_str = f'Loss={loss.item():0.5f} LossMask={loss_mask.item():0.5f} LossDepth={loss_depth.item():05f} Batch_id={batch_idx}'
    if l1_decay > 0:
      pbar_str = f'L1_loss={l1_loss.item():0.3f} %s' % (pbar_str)
    if l2_decay > 0:
      pbar_str = f'L2_loss={l2_loss.item():0.3f} %s' % (pbar_str)

    pbar.set_description(desc= pbar_str)
    

  avg_loss /= len(train_loader)
  avg_acc = 100*correct/processed
  train_losses.append(avg_loss)
  
  # Saving the images
  sample = fg_bg[0:8,:,:,:]
  save_plot(sample.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_Overlayed_Train.jpg")
  #Mask  
  sample = fg_bg_mask[0:8,:,:,:]
  save_plot(sample.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_ActualMask_Train.jpg")
  #Predicted Mask
  output_ = mask_pred[0:8,:,:,:]
  save_plot(output_.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_PredMask_Train.jpg")
  #Depth
  sample = fg_bg_depth[0:8,:,:,:]
  save_plot(sample.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_ActualDepth_Train.jpg")
  #Predicted Depth
  output = depth_pred[0:8,:,:,:]
  save_plot(output.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_PredDepth_Train.jpg")
