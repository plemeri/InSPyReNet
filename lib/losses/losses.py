import torch
import torch.nn.functional as F

def iou_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    return (iou + bce).mean()

def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()

def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()

def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.flatten()
    mask = mask.flatten()

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return (1 - Tversky) ** gamma

def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.flatten()
    mask = mask.flatten()

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return bce + (1 - Tversky) ** gamma

def weighted_tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()
    
    bce = weight * F.binary_cross_entropy_with_logits(pred, mask, reduction='none').flatten()

    pred = torch.sigmoid(pred)       
    pred = pred.flatten()
    mask = mask.flatten()

    #True Positives, False Positives & False Negatives
    TP = (pred * mask)  
    FP = ((1 - mask) * pred)
    FN = (mask * (1 - pred))

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return (bce + (1 - Tversky) ** gamma).mean()

def dweighted_tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()
    
    wbce = weight * F.binary_cross_entropy_with_logits(pred, mask, reduction='none').flatten()

    pred = torch.sigmoid(pred)       
    pred = pred.flatten()
    mask = mask.flatten()

    #True Positives, False Positives & False Negatives
    TP = (pred * mask)  
    FP = ((1 - mask) * pred)
    FN = (mask * (1 - pred))

    wtversky = weight * ((TP + 1) / (TP + alpha * FP + beta * FN + 1))
    return (wbce + (1 - wtversky) ** gamma).mean()