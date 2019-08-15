from __future__ import division
import torch
from torch import nn


def compute_errors(gt, pred):
    abs_diff = 0

    weights = [0.25, 0.25, 0.25, 0.25]

    if type(pred) not in [list, tuple]:
        pred = [pred]
        weights = [1]

    for i, scale_pred in enumerate(pred):

        _, _, hs, ws = scale_pred.size()

        for current_gt, current_pred in zip(gt, scale_pred):
                h, w = current_gt.size()
                scale_gt = nn.functional.interpolate(current_gt.view(1,1,h,w), size=(hs, ws), mode='bilinear', align_corners=False).view(1,hs,ws)
                valid = scale_gt != 0
                abs_diff += weights[i] * torch.mean(torch.abs(scale_gt[valid] - current_pred[valid]))


    return abs_diff


