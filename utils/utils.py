import smtplib
import torch
import yaml
import torch.nn as nn
import cv2
import numpy as np

from easydict import EasyDict as ed
from email.mime.text import MIMEText


def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))


def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample


def patch(x, patch_size=256):
    b, c, h, w = x.shape
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=patch_size // 2)

    patches = unfold(x)
    patches = patches.reshape(
        c, patch_size, patch_size, -1).contiguous().permute(3, 0, 1, 2)

    return patches, (b, c, h, w)


def stitch(patches, target_shape, patch_size=256):
    b, c, h, w = target_shape
    fold = nn.Fold(output_size=(h, w), kernel_size=(
        patch_size,) * 2, stride=patch_size // 2)
    unfold = nn.Unfold(kernel_size=(patch_size,) * 2, stride=patch_size // 2)

    patches = patches.permute(1, 2, 3, 0).reshape(
        b, c * patch_size ** 2, patches.shape[0] // b)

    weight = torch.ones(*target_shape).to(patches.device)
    weight = unfold(weight)

    out = fold(patches) / fold(weight)

    return out


def send_mail(sendto, msg_header, msg_body, user, passwd):
    smtpsrv = "smtp.office365.com"
    smtpserver = smtplib.SMTP(smtpsrv, 587)

    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.login(user, passwd)

    msg = MIMEText(msg_body)
    msg['From'] = user
    msg['To'] = sendto
    msg['Subject'] = msg_header

    smtpserver.sendmail(user, sendto, msg.as_string())
    smtpserver.close()


def debug_tile(out, size=(100, 100)):
    debugs = []
    for debs in out['debug']:
        debug = []
        for deb in debs:
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = (log - log.min()) / (log.max() - log.min())
            log *= 255
            log = log.astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)
