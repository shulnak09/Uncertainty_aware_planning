import torch
import numpy as np
import random
import time

def normalize(vid, mean, std, eps):
    return (vid - mean) / (std+eps)

def unnormalize(vid, mean, std, eps):
    return vid * (std+eps) + mean

def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid
   
def minmax_denormalize(vid, vmin, vmax, scale=2):
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin

class Normalize(object):
    def __init__(self, mean, std, eps=1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std, self.eps)
   
    def inverse_transform(self, vid):
        return unnormalize(vid, self.mean, self.std, self.eps)

class MinMaxNormalize(object):
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid):
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)
   
    def inverse_transform(self, vid):
        return minmax_denormalize(vid, self.datamin, self.datamax, self.scale)