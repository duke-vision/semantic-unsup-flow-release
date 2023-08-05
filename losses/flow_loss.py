import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg
        
            
    def loss_photomatric(self, im1_scaled, im1_recons, vis_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * vis_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * vis_mask1,
                                            im1_scaled * vis_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * vis_mask1,
                                                      im1_scaled * vis_mask1)]
        
        return sum([l.mean() for l in loss]) / (vis_mask1.mean() + 1e-6)

    
    def loss_smooth(self, flow, im1_scaled, sem1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
            
        loss = []
        loss += [func_smooth(flow, im1_scaled, sem1_scaled, edge='image', alpha=self.cfg.edge_aware_alpha)]
        return sum([l.mean() for l in loss])

    
    def get_vis_masks(self, level, pyramid_flows=None):
        # we assume the first vis_mask on level 0 has been initialized
        if level >= len(self.pyramid_vis_mask1):
            for i in range(len(self.pyramid_vis_mask1), level+1):
                _, _, h, w = pyramid_flows[i].size()
                vis_mask1 = F.interpolate(self.pyramid_vis_mask1[0], (h, w), mode='nearest')
                vis_mask2 = F.interpolate(self.pyramid_vis_mask2[0], (h, w), mode='nearest')
                self.pyramid_vis_mask1.append(vis_mask1)
                self.pyramid_vis_mask2.append(vis_mask2)
        
        return self.pyramid_vis_mask1[level], self.pyramid_vis_mask2[level]
                

    def forward(self, pyramid_flows, im1_origin, im2_origin, sem1_origin, sem2_origin, occ_aware=True):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """
        DEVICE = pyramid_flows[0].device
        
        # process data
        B, num_class, H, W = sem1_origin.shape
                        
        # img1_show = (im1_origin.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).reshape((4*256, 832, 3))
        # from utils.semantics_utils import trainId2color
        # sem1_show = trainId2color(sem1_origin.argmax(dim=1).cpu().numpy()).astype(np.uint8).reshape((4*256, 832, 3))
        # import matplotlib.pyplot as plt
        # plt.imsave('_DEBUG_IMG1.png', img1_show)
        # plt.imsave('_DEBUG_SEM1.png', sem1_show)
        
        # generate visibility mask/occlusion estimation
        top_flow = pyramid_flows[0]
        if self.cfg.occ_from_back:
            vis_mask1 = 1 - get_occu_mask_backward(top_flow[:, 2:], th=0.2)
            vis_mask2 = 1 - get_occu_mask_backward(top_flow[:, :2], th=0.2)
        else:
            vis_mask1 = 1 - get_occu_mask_bidirection(top_flow[:, :2], top_flow[:, 2:])
            vis_mask2 = 1 - get_occu_mask_bidirection(top_flow[:, 2:], top_flow[:, :2])
   
        self.pyramid_vis_mask1 = [vis_mask1]
        self.pyramid_vis_mask2 = [vis_mask2]
        scale = min(*top_flow.shape[-2:])

        # compute losses at each level
        pyramid_warp_losses = []
        pyramid_semantic_losses = []
        pyramid_smooth_losses = []
        
        zero_loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)
        
        for i, flow in enumerate(pyramid_flows):

            # resize images to match the size of layer
            b, _, h, w = flow.size()
            im1_scaled, im2_scaled = None, None
            sem1_scaled, sem2_scaled = None, None
            
            # photometric loss
            if self.cfg.w_ph_scales[i] > 0:
                im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
                im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')
                im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
                im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)
                
                if occ_aware:                  
                    vis_mask1, vis_mask2 = self.get_vis_masks(i, pyramid_flows)
                else:
                    vis_mask1 = torch.ones((b, 1, h, w), dtype=torch.float32, device=DEVICE)
                    vis_mask2 = torch.ones((b, 1, h, w), dtype=torch.float32, device=DEVICE)
                    
                loss_warp = self.loss_photomatric(im1_scaled, im1_recons, vis_mask1)
                if self.cfg.with_bk:
                    loss_warp += self.loss_photomatric(im2_scaled, im2_recons, vis_mask2)   
                    loss_warp /= 2.
                pyramid_warp_losses.append(loss_warp)
                
            else:
                pyramid_warp_losses.append(zero_loss)
                
            # smoothness loss
            if self.cfg.w_smooth > 0 and self.cfg.w_sm_scales[i] > 0:
                if im1_scaled is None:
                    im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
                    im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')
                    
                if sem1_scaled is None:
                    sem1_scaled = F.interpolate(sem1_origin, (h, w), mode='area')
                    sem2_scaled = F.interpolate(sem2_origin, (h, w), mode='area')
                
                loss_smooth = self.loss_smooth(flow[:, :2] / scale, im1_scaled, sem1_scaled)                
                if self.cfg.with_bk:
                    loss_smooth += self.loss_smooth(flow[:, 2:] / scale, im2_scaled, sem2_scaled)
                    loss_smooth /= 2.
                pyramid_smooth_losses.append(loss_smooth)
            
            else:
                pyramid_smooth_losses.append(zero_loss)
                
            
            # debug: print to see
            '''
            import numpy as np
            from utils.flow_utils import flow_to_image
            from utils.semantics_utils import trainId2color
            import matplotlib.pyplot as plt
            
            img1_show = (im1_scaled.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            img2_show = (im2_scaled.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            sem1_show = trainId2color(sem1_scaled.argmax(dim=1).cpu().numpy()).astype(np.uint8)
            sem2_show = trainId2color(sem2_scaled.argmax(dim=1).cpu().numpy()).astype(np.uint8)
            
            flow12_numpy = flow[:, :2].detach().cpu().numpy().transpose(0, 2, 3, 1)
            flow12_show = []
            for f in flow12_numpy:
                flow12_show.append(flow_to_image(f))
            flow12_show = np.stack(flow12_show)
            
            flow21_numpy = flow[:, 2:].detach().cpu().numpy().transpose(0, 2, 3, 1)
            flow21_show = []
            for f in flow21_numpy:
                flow21_show.append(flow_to_image(f))
            flow21_show = np.stack(flow21_show)
            
            vis1_show = (vis_mask1.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            vis2_show = (vis_mask2.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            img1_warp_show = (im1_recons.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            img2_warp_show = (im2_recons.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            sem1_warp_show = trainId2color(sem1_recons.detach().argmax(dim=1).cpu().numpy()).astype(np.uint8)
            sem2_warp_show = trainId2color(sem2_recons.detach().argmax(dim=1).cpu().numpy()).astype(np.uint8)   
            
            ternary12, ternary21, sem12, sem21 = TEMP[-4:]
            ternary12_show = (ternary12.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            ternary21_show = (ternary21.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            sem12_show = ((sem12/2).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            sem21_show = ((sem21/2).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
                   
            all_show = np.concatenate((np.concatenate((img1_show, img2_show, sem1_show, sem2_show), axis=1), 
                                       np.concatenate((flow12_show, flow21_show, vis1_show, vis2_show), axis=1),
                                       np.concatenate((img1_warp_show, img2_warp_show, sem1_warp_show, sem2_warp_show), axis=1),
                                       np.concatenate((ternary12_show, ternary21_show, sem12_show, sem21_show), axis=1)),
                                      axis=2)
            b, h, w, c = all_show.shape
            all_show = np.concatenate((all_show[:, :, :w//2, :], all_show[:, :, w//2:, :]), axis=1)
            #all_show = all_show.reshape((b*h, w, c))
            all_show = all_show[0]
            
            import IPython; IPython.embed(); exit()
            plt.imsave('_DEBUG_DEMO_{}.png'.format(i), all_show)
            '''
            
            '''
            if i == 0:  # for analysis
                self.l_ph_0 = loss_warp
                self.l_ph_L1_map_0 = (im1_scaled - im1_recons).abs().mean(dim=1)
            '''
        
        # aggregate losses      
        pyramid_warp_losses = [l * w for l, w in zip(pyramid_warp_losses, self.cfg.w_ph_scales)]
        pyramid_smooth_losses = [l * w for l, w in zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        l_ph = sum(pyramid_warp_losses)
        l_sm = sum(pyramid_smooth_losses)
        
        total_loss = l_ph + self.cfg.w_smooth * l_sm
        
        return total_loss, l_ph, l_sm, pyramid_flows[0].abs().mean()

