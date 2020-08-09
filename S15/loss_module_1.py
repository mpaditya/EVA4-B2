# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from kornia.losses import SSIM
from kornia.filters import SpatialGradient
from torch.nn import functional as F


#def inverse_huber_loss(target, output):
#    absdiff = torch.abs(output-target)
#    C = 0.2*torch.max(absdiff).item()
#    return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
def DiceLoss (inputs, targets, smooth=1e-6):
        
#        #comment out if your model contains a sigmoid or equivalent activation layer
#        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
                
        return 1 - dice
    
def IoULoss (inputs, targets, smooth=1e-6):
        
#        #comment out if your model contains a sigmoid or equivalent activation layer
#        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
def IoU_BCELoss (targets, inputs, smooth=1e-6):
        
#        #comment out if your model contains a sigmoid or equivalent activation layer
#        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        iou_loss = 1 - IoU
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        IoU_BCE = BCE + iou_loss
        
        return IoU_BCE

def Dice_BCELoss (targets, inputs, smooth=1e-6):
        
#        #comment out if your model contains a sigmoid or equivalent activation layer
#        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        dice_loss = 1 - dice
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        IoU_BCE = BCE + dice_loss
        
        return IoU_BCE
    
def edge_loss(target, output):
	gt_pred = SpatialGradient()(output)
	assert(gt_pred.ndim == 5)
	assert(gt_pred.shape[2] == 2)
	dy_pred = gt_pred[:,:,0,:,:]
	dx_pred = gt_pred[:,:,1,:,:]
	gt_true = SpatialGradient()(target)
	dy_true = gt_true[:,:,0,:,:]
	dx_true = gt_true[:,:,1,:,:]
	l_edge = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
	return l_edge

def loss_mask_function(target, output):
	# IoU loss	
#    l_iou = IoULoss(target, output)
#    l_dice = DiceLoss(target, output)
	
#    # BCE + IoU loss	
   l_iou = IoU_BCELoss(target, output)
#    s = f"(d{l_iou.item():0.3f})"
#    s = f"(d{l_dice.item():0.3f})"
   loss = l_iou
#    loss = F.binary_cross_entropy_with_logits(output, target)
   s = f"(d{l_iou:0.3f})"
   return loss, s

def loss_depth_function(target, output, w_ssim=1.0, w_edge=1.0, w_depth=1.0):
	# Structural similarity (SSIM) index
	# 1 - ssim_index is computed internally, within SSIM()
	l_ssim = SSIM(3, reduction="mean")(output, target)

	# Edges
	l_edge = edge_loss(output, target)

	# Point-wise depth
	l_depth = nn.L1Loss()(output, target)
	# l_huber = inverse_huber_loss(target, output)
	# l_mse = nn.MSELoss()(output*10, target*10)
	# l_bce = nn.BCEWithLogitsLoss()(output, target)
	s = f"(d{l_depth.item():0.3f},s{l_ssim.item():0.3f},e{l_edge.item():0.3f})"
	loss = (w_ssim * l_ssim) + (w_depth * l_depth) + (w_edge * l_edge)
	return loss, s