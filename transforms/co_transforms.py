import numbers
import random
import numpy as np
# from scipy.misc import imresize
from skimage.transform import resize as imresize
import scipy.ndimage as ndimage


def get_co_transforms(aug_args):
    transforms = []
    # for cityscapes, where we crop to remove the car logo
    if 'crop_bottom' in aug_args and aug_args.crop_bottom > 0:
        transforms.append(CropBottom(aug_args.crop_bottom))
    if aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop))
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.swap:
        transforms.append(RandomSwap())
    return Compose(transforms)


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, semantics, target):
        for t in self.co_transforms:
            input, semantics, target = t(input, semantics, target)
        return input, semantics, target

    
class CropBottom(object):
    def __init__(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def __call__(self, inputs, semantics, target):
        h, w, _ = inputs[0].shape
        new_h = int(h * (1 - self.crop_ratio))

        inputs = [img[:new_h] for img in inputs]
        semantics = [sem[:new_h] for sem in semantics]
        if 'mask' in target:
            target['mask'] = target['mask'][:new_h]
        if 'flow' in target:
            target['flow'] = target['flow'][:new_h]
        return inputs, semantics, target
    
    
class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, semantics, target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [img[y1: y1 + th, x1: x1 + tw] for img in inputs]
        semantics = [sem[y1: y1 + th, x1: x1 + tw] for sem in semantics]
        if 'mask' in target:
            target['mask'] = target['mask'][y1: y1 + th, x1: x1 + tw]
        if 'flow' in target:
            target['flow'] = target['flow'][y1: y1 + th, x1: x1 + tw]
        return inputs, semantics, target


class RandomSwap(object):
    def __call__(self, inputs, semantics, target):
        n = len(inputs)
        if random.random() < 0.5:
            inputs = inputs[::-1]
            semantics = semantics[::-1]
            if 'mask' in target:
                target['mask'] = target['mask'][::-1]
            if 'flow' in target:
                raise NotImplementedError("swap cannot apply to flow")
        return inputs, semantics, target


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, semantics, target):
        if random.random() < 0.5:
            inputs = [np.fliplr(im) for im in inputs]
            semantics = [np.fliplr(sem) for sem in semantics]
            if 'mask' in target:
                target['mask'] = [np.copy(np.fliplr(mask)) for mask in target['mask']]
            if 'flow' in target:
                for i, flo in enumerate(target['flow']):
                    flo = np.copy(np.fliplr(flo))
                    flo[:, :, 0] *= -1
                    target['flow'][i] = flo
        return inputs, semantics, target