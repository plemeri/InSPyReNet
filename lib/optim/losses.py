import torch
import torch.nn.functional as F

def bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    return bce.mean()

def lap_loss(pred, mask):
    pred = torch.tanh(pred)
    return F.mse_loss(pred, mask)

def weighted_tversky_bce_loss_with_logits(pred, mask, alpha=0.5, beta=0.5, gamma=2):
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

def weighted_tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()
    
    bce = weight * F.binary_cross_entropy_with_logits(pred, mask, reduction='none').flatten()

    pred = pred.flatten()
    mask = mask.flatten()

    #True Positives, False Positives & False Negatives
    TP = (pred * mask)  
    FP = ((1 - mask) * pred)
    FN = (mask * (1 - pred))

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return (bce + (1 - Tversky) ** gamma).mean()
