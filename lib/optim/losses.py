import torch
import torch.nn.functional as F

# def bce_iou_loss(pred, mask):
#     weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

#     bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
#     pred = torch.sigmoid(pred)
#     inter = pred * mask
#     union = pred + mask
#     iou = 1 - (inter + 1) / (union - inter + 1)

#     weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
#     weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

#     return (weighted_bce + weighted_iou).mean()


# def weighted_tversky_bce_loss_with_logits(pred, mask, alpha=0.5, beta=0.5, gamma=2):
#     weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     weight = weight.flatten()
    
#     bce = weight * F.binary_cross_entropy_with_logits(pred, mask, reduction='none').flatten()

#     pred = torch.sigmoid(pred)       
#     pred = pred.flatten()
#     mask = mask.flatten()

#     #True Positives, False Positives & False Negatives
#     TP = (pred * mask)  
#     FP = ((1 - mask) * pred)
#     FN = (mask * (1 - pred))

#     Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

#     return (bce + (1 - Tversky) ** gamma).mean()

# def weighted_tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
#     weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     weight = weight.flatten()
    
#     bce = weight * F.binary_cross_entropy(pred, mask, reduction='none').flatten()

#     pred = pred.flatten()
#     mask = mask.flatten()

#     #True Positives, False Positives & False Negatives
#     TP = (pred * mask)  
#     FP = ((1 - mask) * pred)
#     FN = (mask * (1 - pred))

#     Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

#     return (bce + (1 - Tversky) ** gamma).mean()


# def tversky_loss_with_logits(pred, mask, alpha=0.5, beta=0.5, gamma=2):
#     pred = pred.flatten()
#     mask = mask.flatten()

#     #True Positives, False Positives & False Negatives
#     TP = (pred * mask)  
#     FP = ((1 - mask) * pred)
#     FN = (mask * (1 - pred))

#     Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

#     return (1 - Tversky) ** gamma

def bce_loss(pred, mask, reduction='none'):
    bce = F.binary_cross_entropy(pred, mask, reduction=reduction)
    return bce

def weighted_bce_loss(pred, mask, reduction='none'):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()
    
    bce = weight * bce_loss(pred, mask, reduction='none').flatten()
    
    if reduction == 'mean':
        bce = bce.mean()
    
    return bce

def iou_loss(pred, mask, reduction='none'):
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    if reduction == 'mean':
        iou = iou.mean()

    return iou

def tversky_loss(pred, mask, alpha=0.5, beta=0.5, reduction='none'):
    pred = pred.flatten()
    mask = mask.flatten()

    #True Positives, False Positives & False Negatives
    TP = (pred * mask)  
    FP = ((1 - mask) * pred)
    FN = (mask * (1 - pred))

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)
    
    if reduction == 'mean':
        Tversky = Tversky.mean()
    
    return Tversky

def focal_tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2, reduction='none'):
    Tversky = tversky_loss(pred, mask, alpha=alpha, beta=beta, reduction='none')
    FTversky = (1 - Tversky) ** gamma
    
    if reduction == 'mean':
        FTversky = FTversky.mean()
    
    return FTversky    

def bce_loss_with_logits(pred, mask, reduction='none'):
    return bce_loss(torch.sigmoid(pred), mask, reduction=reduction)

def weighted_bce_loss_with_logits(pred, mask, reduction='none'):
    return weighted_bce_loss(torch.sigmoid(pred), mask, reduction=reduction)

def iou_loss_with_logits(pred, mask, reduction='none'):
    return iou_loss(torch.sigmoid(pred), mask, reduction=reduction)
    
def tversky_loss_with_logits(pred, mask, alpha=0.5, beta=0.5, reduction='none'):
    return tversky_loss(torch.sigmoid(pred), mask, alpha=alpha, beta=beta, reduction=reduction)

def focal_tversky_loss_with_logits(pred, mask, alpha=0.5, beta=0.5, gamma=2, reduction='none'):
    return focal_tversky_loss(torch.sigmoid(pred), mask, alpha=alpha, beta=beta, gamma=gamma, reduction=reduction)