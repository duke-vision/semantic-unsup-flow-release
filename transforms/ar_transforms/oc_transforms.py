import numpy as np
import torch
# from skimage.color import rgb2yuv
import cv2

from utils.warp_utils import flow_warp


def random_crop(img, flow, occ_mask, crop_sz):
    """

    :param img: Nx6xHxW
    :param flows: n * [Nx2xHxW]
    :param occ_masks: n * [Nx1xHxW]
    :param crop_sz:
    :return:
    """
    _, _, h, w = img.size()
    c_h, c_w = crop_sz

    if c_h == h and c_w == w:
        return img, flow, occ_mask

    x1 = np.random.randint(0, w - c_w)
    y1 = np.random.randint(0, h - c_h)
    img = img[:, :, y1:y1 + c_h, x1: x1 + c_w]
    flow = flow[:, :, y1:y1 + c_h, x1: x1 + c_w]
    occ_mask = occ_mask[:, :, y1:y1 + c_h, x1: x1 + c_w]

    return img, flow, occ_mask


def semantic_connected_components(semseg, class_indices=[13], width_range=(30, 200), height_range=(30, 100)):
    '''
        Input:
            semsegs: Onehot semantic segmentations of size [c, H, W]
            class_indices: A list of the indices of the classes of interest.
                           For example, [car_idx] or [sign_idx, pole_idx, traffic_light_idx]
            width_range, height_range: The width and height ranges for the objects of interest.
        Ouput:
            list of masks for cars in the size range
    '''
     
    curr_sem = semseg[class_indices].sum(dim=0)
    curr_sem = (curr_sem[:, :, None] * 255).numpy().astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(curr_sem)
    
    sem_list = []
    # 0 is background, so ignore them.
    for i in range(1, num_labels):
        curr_obj = (labels == i)
        hs, ws = np.where(curr_obj)
        h_len, w_len = np.max(hs) - np.min(hs), np.max(ws) - np.min(ws)
        if (height_range[0] < h_len < height_range[1]) and (width_range[0] < w_len < width_range[1]):
            if curr_obj.sum() / ((h_len+1) * (w_len+1)) > 0.6:  # filter some wrong car estimates or largely occluded cars
                sem_list.append(curr_obj)

    return sem_list


def find_semantic_group(semseg, class_indices=[5, 6, 7], win_width=200):
    curr_sem = semseg[class_indices].sum(dim=0).numpy()
    freq = curr_sem.mean(axis=0)  # 1d frequency
    freq_win = np.convolve(freq, np.ones(win_width) / win_width, mode='valid')  # find the most frequent window

    if max(freq_win) > 0.1:
        # optimal window: [left, right)
        left = np.argmax(freq_win)
        right = left + win_width
        curr_sem[:, :left] = 0
        curr_sem[:, right:] = 0
        return curr_sem
    
    else:
        return None
        

def add_fake_object(input_dict, return_new_occ=True):
    
    # prepare input
    img1_ot = input_dict['img1_tgt']
    img2_ot = input_dict['img2_tgt']
    sem1_ot = input_dict['sem1_tgt']
    sem2_ot = input_dict['sem2_tgt']
    flow_ot = input_dict['flow_tgt']
    noc_ot = input_dict['noc_tgt']
    
    img = input_dict['img_src']
    sem = input_dict['sem_src']
    obj_mask = input_dict['obj_mask']
    motion = input_dict['motion'][:, :, None, None]
    
    b, _, h, w = img1_ot.shape
                    
    # add object to frame 1
    img1_ot = obj_mask * img + (1 - obj_mask) * img1_ot
    sem1_ot = obj_mask * sem + (1 - obj_mask) * sem1_ot
    
    # add object to frame 2
    new_obj_mask = flow_warp(obj_mask, - motion.repeat(1, 1, h, w), pad='zeros')
    new_img = flow_warp(img, - motion.repeat(1, 1, h, w), pad='border')
    new_sem = flow_warp(sem, - motion.repeat(1, 1, h, w), pad='border')
    img2_ot = new_obj_mask * new_img + (1 - new_obj_mask) * img2_ot
    sem2_ot = new_obj_mask * new_sem + (1 - new_obj_mask) * sem2_ot
    
    if return_new_occ: # use the original forward flow (not the modified one!) to warp newly occluded regions
        new_occ_mask = flow_warp(new_obj_mask, flow_ot, pad='zeros')
    
    # change flow
    flow_ot = obj_mask * motion + (1 - obj_mask) * flow_ot
    noc_ot = torch.max(noc_ot, obj_mask)  # where we are confident about flow_ot
    
    if return_new_occ:
        return img1_ot, img2_ot, sem1_ot, sem2_ot, flow_ot, noc_ot, new_obj_mask, new_occ_mask
    else:
        return img1_ot, img2_ot, sem1_ot, sem2_ot, flow_ot, noc_ot, new_obj_mask
    