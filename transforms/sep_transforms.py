import numpy as np
import torch
import cv2

from sklearn.preprocessing import OneHotEncoder
from utils.semantics_utils import num_class

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, imgs_sems_tuple):
        imgs, sems = imgs_sems_tuple
        
        imgs = [torch.from_numpy(img.transpose((2, 0, 1))).float() for img in imgs]
        sems = [torch.from_numpy(sem.transpose((2, 0, 1))).float() for sem in sems]
        
        return imgs, sems

    
class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, imgs_sems_tuple):
        imgs, sems = imgs_sems_tuple
        
        imgs = [cv2.resize(img, (self.new_w, self.new_h)) for img in imgs]
        sems = [cv2.resize(sem, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST) for sem in sems]
        
        return imgs, sems


class OneHotSemantics(object):
    def __init__(self):
        self.enc = OneHotEncoder(sparse=False)
        self.enc.fit(np.array(range(num_class))[:, None])

    def __call__(self, imgs_sems_tuple):
        imgs, sems = imgs_sems_tuple
        sems = [self.enc.transform(sem.reshape((-1, 1))).reshape((*sem.shape, -1)) for sem in sems]
        
        return imgs, sems