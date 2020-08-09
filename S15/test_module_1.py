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

def iou(gt, pred):
    
    """Calculate Intersection over Union.
    
    Args:
        label (torch.Tensor): Ground truth.
        prediction (torch.Tensor): Prediction.
    
    Returns:
        IoU
    """
    # Remove channel dimension - Convert BATCH x 1 x H x W => BATCH x H x W
    gt = gt.squeeze(1)
    pred = pred.squeeze(1)
    
#    #comment out if your model contains a sigmoid or equivalent activation layer
#    pred = F.sigmoid(pred)
#    gt = F.sigmoid(gt)
    
    intersection = (pred * gt).float().sum(2).sum(1)  # Will be zero if Truth=0 or Prediction=0
    union = (pred + gt).float().sum(2).sum(1) - intersection   # Will be zero if both are 0
    
    SMOOTH = 1e-6
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return iou.float().mean().item()  # If you are interested in average across the batch


def test(model, device, test_loader, test_losses, epoch=0, save_img=True):
    model.eval()
    test_loss = 0
    mask_loss = 0
    depth_loss = 0
    eval_m = 0
    eval_d = [0,0,0,0]
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            bg = data["f1"]
            fg_bg = data["f2"]
            fg_bg_mask = data["f3"].to(device)
            fg_bg_depth = data["f4"].to(device)
            inp = torch.cat([bg,fg_bg], dim=1)
            inp = inp.to(device)
            mask_pred, depth_pred = model(inp)
            # Calculate loss
            loss_mask, mstr = loss_mask_function(output=mask_pred, target=fg_bg_mask,
                                    w_depth=1.0, w_ssim=1.0, w_edge=1.0)
            loss_depth, dstr = loss_depth_function(output=depth_pred, target=fg_bg_depth,
                                    w_depth=1.0, w_ssim=1.0, w_edge=1.0)
            loss = loss_mask + loss_depth
            test_loss += loss.item()
            mask_loss += loss_mask.item()
            depth_loss += loss_depth.item()


#            pem = evaluate(fg_bg_mask, mask_pred)
            pem = iou(fg_bg_mask, mask_pred)
            ped = evaluate(fg_bg_depth, depth_pred)
            for i in range(len(ped)):
                eval_d[i] += ped[i]
                
            eval_m += pem

    ##### SAVING THE IMAGES
    #Overlayed
    sample = fg_bg[0:8,:,:,:]
    save_plot(sample.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_Overlayed_Test.jpg")
    #Mask
    sample = fg_bg_mask[0:8,:,:,:]
    save_plot(sample.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_ActualMask_Test.jpg")
    #Predicted Mask
    output_ = mask_pred[0:8,:,:,:]
    save_plot(output_.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_PredMask_Test.jpg")
    #Depth
    sample = fg_bg_depth[0:8,:,:,:]
    save_plot(sample.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_ActualDepth_Test.jpg")
    #Predicted Depth
    output = depth_pred[0:8,:,:,:]
    save_plot(output.detach().cpu(), f"/content/gdrive/My Drive/S15/plots/{epoch}_PredDepth_Test.jpg")
    
    test_loss /= len(test_loader)
    mask_loss /= len(test_loader)
    depth_loss /= len(test_loader)
    test_losses.append(test_loss)

    eval_m /= len(test_loader)
    for i in range(len(ped)):
        eval_d[i] /= len(test_loader)
    
    print('\n Test set: Average loss: {:.4f}, Average MaskLoss: {:.4f}, Average DepthLoss: {:.4f}\n'.format(
        test_loss, mask_loss, depth_loss))
    
    print("{}: {:>10}, {:>10}, {:>10}, {:>10}".format(
        "Metric",'t<1.25', 't<1.25^2', 't<1.25^3', 'rms'))
    
    print("{}: {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}  \n".format(
        "Depth ",eval_d[0],eval_d[1],eval_d[2],eval_d[3]))
    
    print("{}: {:>10}".format(
        "Metric",'IoU'))
    
    print("{}: {:10.4f}".format(
        "Mask  ",eval_m))


def evaluate(gt, pred):
    rmse = torch.sqrt(torch.nn.MSELoss()(gt, pred))
    # abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    
    # While calculating t<1.25, we want the ratio of pixels to within a threshold
    # like 1.25. But if the value of pixel is less than 0.1 then even though the
    # pixel values are close the ratio scale changes
    # For ex, 0.00001 and 0.000001 are very close and we want them to contribute
    # positively for our accuracy but the ratio is 10 which reduces the accuracy.
    # So we clamp the tensors to 0.1 and 1   
    gt = torch.clamp(gt, min=0.1, max=1)
    pred = torch.clamp(pred, min=0.1, max=1)

    thresh = torch.max((gt / pred), (pred / gt))

    a1 = (thresh < 1.25   ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    return a1.item(), a2.item(), a3.item(), rmse.item()