import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve

EPS = np.finfo(np.float64).eps

def Object(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2.0 * x / (x ** 2 + 1 + sigma_x + EPS)

    return score

def S_Object(pred, gt):
    pred_fg = pred.copy()
    pred_fg[gt != 1] = 0.0
    O_fg = Object(pred_fg, gt)
    
    pred_bg = (1 - pred.copy())
    pred_bg[gt == 1] = 0.0
    O_bg = Object(pred_bg, 1-gt)

    u = np.mean(gt)
    Q = u * O_fg + (1 - u) * O_bg

    return Q

def centroid(gt):
    if np.sum(gt) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    
    else:
        x, y = np.where(gt == 1)
        return int(np.mean(x).round()), int(np.mean(y).round())

def divide(gt, x, y):
    LT = gt[:x, :y]
    RT = gt[x:, :y]
    LB = gt[:x, y:]
    RB = gt[x:, y:]

    w1 = LT.size / gt.size
    w2 = RT.size / gt.size
    w3 = LB.size / gt.size
    w4 = RB.size / gt.size

    return LT, RT, LB, RB, w1, w2, w3, w4

def ssim(pred, gt):
    x = np.mean(pred)
    y = np.mean(gt)
    N = pred.size

    sigma_x2 = np.sum((pred - x) ** 2 / (N - 1 + EPS))
    sigma_y2 = np.sum((gt - y) ** 2 / (N - 1 + EPS))

    sigma_xy = np.sum((pred - x) * (gt - y) / (N - 1 + EPS))

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + EPS)
    elif alpha == 0 and beta == 0:
        Q = 1
    else:
        Q = 0
    
    return Q

def S_Region(pred, gt):
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide(gt, x, y)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, x, y)

    Q1 = ssim(pred1, gt1)
    Q2 = ssim(pred2, gt2)
    Q3 = ssim(pred3, gt3)
    Q4 = ssim(pred4, gt4)

    Q = Q1 * w1 + Q2 * w2 + Q3 * w3 + Q4 * w4

    return Q

def StructureMeasure(pred, gt):
    y = np.mean(gt)

    if y == 0:
        x = np.mean(pred)
        Q = 1 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        alpha = 0.5
        Q = alpha * S_Object(pred, gt) + (1 - alpha) * S_Region(pred, gt)
        if Q < 0:
            Q = 0
    
    return Q

def thresholdBased_HR_FR(pred, thresholds, gt):
    gtPxlNum = np.sum(gt)
    totalNum = gt.size

    cthresholds = np.zeros(thresholds.size + 1)
    cthresholds[:-1] += thresholds[::-1]
    cthresholds[1:] += thresholds[::-1]
    cthresholds[0] += 2 * thresholds[::-1][0] - thresholds[::-1][1]
    cthresholds[-1] += 2 * thresholds[::-1][-1] - thresholds[::-1][-2]
    cthresholds /= 2
    
    targetHist, _ = np.histogram(pred[gt == 1], cthresholds)
    nontargetHist, _ = np.histogram(pred[gt != 1], cthresholds)

    targetHist = targetHist[::-1]
    nontargetHist = nontargetHist[::-1]

    targetHist = np.cumsum(targetHist)
    nontargetHist = np.cumsum(nontargetHist)

    Pre = targetHist / (targetHist + nontargetHist + EPS)
    Recall = targetHist / (gtPxlNum + EPS)
    TPR = Recall
    FPR = nontargetHist / (totalNum - gtPxlNum)
    IoU = targetHist / (gtPxlNum + nontargetHist)
    hitRate = targetHist / (gtPxlNum + EPS)
    falseAlarm = 1 - ((totalNum - gtPxlNum) - nontargetHist) / (totalNum - gtPxlNum)

    return IoU, TPR, FPR, Pre, Recall, hitRate, falseAlarm

def CalMAE(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    fgPixels = pred[gt == 1]
    fgErrSum = fgPixels.size - np.sum(fgPixels)
    bgErrSum = np.sum(pred[gt != 1])
    mae = (fgErrSum + bgErrSum) / gt.size
    return mae

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def wFmeasure_calu(pred, gt):
    if np.max(gt) == 0:
        return 0

    E = np.abs(pred - gt)
    dst, idst = distance_transform_edt(1 - gt, return_indices=True)

    K = fspecial_gauss(7, 5)
    Et = E.copy()
    Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
    EA = convolve(Et, K, mode='nearest')
    MIN_E_EA = E.copy()
    MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

    B = np.ones_like(gt)
    B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
    Ew = MIN_E_EA * B

    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt != 1])

    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + EPS)
    Q = 2 * R * P / (R + P + EPS)

    return Q

def AlignmentTerm(pred, gt):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    align_pred = pred - mu_pred
    align_gt = gt - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + EPS)
    
    return align_mat

def EnhancedAlighmentTerm(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced

def Emeasure_calu(pred, gt):
    thd = 0.5
    pred = (pred >= thd)

    if np.sum(gt) == 0:
        enhanced_mat = 1.0 - pred
    elif np.sum(1 - gt) == 0:
        enhanced_mat = pred.copy()
    else:
        align_mat = AlignmentTerm(pred, gt)
        enhanced_mat = EnhancedAlighmentTerm(align_mat)
    
    score = np.sum(enhanced_mat) / (gt.size - 1 + EPS)
    return score
