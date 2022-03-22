import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import re
import random
import math

from easydict import EasyDict as ed

import torch.nn as nn
import torch.nn.functional as F

class TOJIT(nn.Module):
    def __init__(self, model, scale=[0.5, 2]):
        super(TOJIT, self).__init__()
        self.model = model
        
    def cuda(self):
        self.model = self.model.cuda()
        return self
        
    def forward(self, sample):
        x = sample['image']
        b, c, h, w = x.shape
        
        out = self.model({'image': x})
        return {'pred': out['pred']}

def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)

def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    return cfg

def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample

def to_numpy(pred, shape):
    pred = F.interpolate(pred, shape, mode='bilinear', align_corners=True)
    pred = pred.data.cpu()
    pred = pred.numpy().squeeze()
    return pred

def debug_tile(deblist, size=(100, 100)):
    debugs = []
    for debs in deblist:
        debug = []
        for deb in debs:
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = ((log - log.min()) / (log.max() - log.min()) * 255).astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)

def unfold(x, patch_size=256):
    b, c, h, w = x.shape
    stride = patch_size // 2
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=stride)

    patches = unfold(x)
    patches = patches.reshape(
        c, patch_size, patch_size, -1).contiguous().permute(3, 0, 1, 2)

    return patches, (b, c, h, w)


def fold(patches, target_shape, patch_size=256):
    b, c, h, w = target_shape
    stride = patch_size // 2
    fold = nn.Fold(output_size=(h, w), kernel_size=(
        patch_size,) * 2, stride=stride)
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=stride)

    patches = patches.permute(1, 2, 3, 0).reshape(
        b, c * patch_size ** 2, patches.shape[0] // b)

    weight = torch.ones(*target_shape).to(patches.device)
    weight = unfold(weight)

    out = fold(patches) # / fold(weight)

    return out


def get_random_structure(size):
    # The provided model is trained with 
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target = 0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate. 
    # input: np array of size [H,W] image
    # output: same shape as input
    
    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #only modified contours is needed actually. 
    sampled_contours = []   
    modified_contours = [] 

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        #remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)
        
        idx_dist = []
        for i in range(number_of_vertices-number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
            
        idx_dist = sorted(idx_dist, key=lambda x:x[1])
        
        remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]
        
       #remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
        contour = new_contour
        

        #sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            #modify contours
            for idx, coor in enumerate(modified_contour):

                change = np.random.normal(0,move_rate) # 0.1 means change position of vertex to 10 percent farther from center
                x,y = coor[0]
                new_x = x + (x-center[0]) * change
                new_y = y + (y-center[1]) * change

                modified_contour[idx] = [new_x,new_y]
        modified_contours.append(modified_contour)
        

    #draw boundary
    gt = np.copy(image)
    image = np.zeros_like(image)

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    image = perturb_seg(image, iou_target)
    
    return image

if __name__ == "__main__":
    x = torch.rand(4, 3, 576, 576)