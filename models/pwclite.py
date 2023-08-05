import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp
from .correlation_package.correlation import Correlation
# from .correlation_native import Correlation

from utils.semantics_utils import num_class

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x, _):
        feature_pyramid = [x]
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]

    
class FeatureExtractorSem(nn.Module):
    def __init__(self, num_chs, sem_num_chs):
        super(FeatureExtractorSem, self).__init__()
        self.num_chs = num_chs
        self.sem_num_chs = sem_num_chs
        self.convs = nn.ModuleList()
        self.sem_convs = nn.ModuleList()
        
        for l, (ch_in, ch_out) in enumerate(zip(sem_num_chs[:-1], sem_num_chs[1:])):  # semantic branch
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.sem_convs.append(layer)        
        
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            if l == len(sem_num_chs) - 1:  # merge two branches here
                ch_in += sem_num_chs[l]
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x, sem):   
        feature_pyramid = [torch.cat((x, sem), dim=1)]
        
        for l in range(len(self.convs)):
            if l < len(self.sem_convs):  # start with separate branch
                sem = self.sem_convs[l](sem)
                x = self.convs[l](x)
                feature_pyramid.append(torch.cat((x, sem), dim=1))
            elif l == len(self.sem_convs):  # merging
                x = self.convs[l](torch.cat((x, sem), dim=1))
                feature_pyramid.append(x)
            else: # merged
                x = self.convs[l](x)
                feature_pyramid.append(x)
                
        return feature_pyramid[::-1]
    
    
class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)
        
    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)
  
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow

    
class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8)
        )
        self.flow_head = nn.Sequential(
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )
   
    def forward(self, x):
        feat = self.convs(x)
        flow = self.flow_head(feat)
        return flow, feat

    
class UpFlowNetwork(nn.Module):
    def __init__(self, ch_in=96, scale_factor=4):
        super(UpFlowNetwork, self).__init__()       
        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, scale_factor ** 2 *9, 3, 1, 1)
        )

    # adapted from https://github.com/princeton-vl/RAFT/blob/aac9dd54726caf2cf81d8661b07663e220c5586d/core/raft.py#L72
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)  
    
    def forward(self, flow, feat):
        # scale mask to balence gradients   
        up_mask = .25 * self.convs(feat)
        return self.upsample_flow(flow, up_mask)
    

class PWCLite(nn.Module):
    def __init__(self, cfg):
        super(PWCLite, self).__init__()
        self.cfg = cfg
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        if not hasattr(cfg, "sem_enc_layers") or cfg.sem_enc_layers is None: # no semantic input in the encoder
            self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        else:
            sem_num_chs = [19, ] + self.num_chs[1:(cfg.sem_enc_layers + 1)]
            self.feature_pyramid_extractor = FeatureExtractorSem(self.num_chs, sem_num_chs)
            for i in range(cfg.sem_enc_layers+1):
                self.num_chs[i] += sem_num_chs[i]
            
        self.upsample = cfg.upsample
        self.reduce_dense = cfg.reduce_dense

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + (self.dim_corr + 2)

        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)
        else:
            self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.flow_estimators.feat_dim + 2)
        
        if hasattr(self.cfg, "learned_upsampler") and self.cfg.learned_upsampler:
            self.output_flow_upsampler = UpFlowNetwork(ch_in=96, scale_factor=4)
            
        self.conv_1x1 = nn.ModuleList([conv(self.num_chs[-1], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(self.num_chs[-2], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(self.num_chs[-3], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(self.num_chs[-4], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(self.num_chs[-5], 32, kernel_size=1, stride=1, dilation=1)])
        
#         import IPython; IPython.embed();
            
    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    
    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)      
        

    def forward_one_way(self, x1_pyramid, x2_pyramid, sem1_full, sem2_full):
        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
           
        # compute the 1by1 features that have the same number of channels for recurrent decoder
        x1_1by1 = []
        for l in range(len(self.conv_1x1)):
            x1_1by1.append(self.conv_1x1[l](x1_pyramid[l]))

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):             
            
            # warping
            if l > 0:
                flow = F.interpolate(flow * 2, scale_factor=2, mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, flow)
            else:
                x2_warp = x2

            # correlation
            out_corr = self.corr(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)
            
            x_intm, flow_res = self.flow_estimators(torch.cat([out_corr_relu, x1_1by1[l], flow], dim=1))
            flow = flow + flow_res
        
            flow_fine, up_feat = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            if self.upsample:
                if hasattr(self.cfg, "learned_upsampler") and self.cfg.learned_upsampler:
                    flow_up = self.output_flow_upsampler(flow, up_feat)
                else:
                    flow_up = F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True)
                    
                flows.append(flow_up)
            else:
                flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break
        
        return flows[::-1]


    def forward(self, img1, img2, sem1, sem2, with_bk=False):
        feat1_pyramid = self.feature_pyramid_extractor(img1, sem1)
        feat2_pyramid = self.feature_pyramid_extractor(img2, sem2)

        res_dict = {}
        res_dict['flows_fw'] = self.forward_one_way(feat1_pyramid, feat2_pyramid, sem1, sem2)
        if with_bk:
            res_dict['flows_bw'] = self.forward_one_way(feat2_pyramid, feat1_pyramid, sem2, sem1)            
                
        return res_dict

