# credits: adapted from https://github.com/anuragranj/cc/blob/master/test_flow.py

import torch
import imageio

from torchvision import transforms

import os
from path import Path
from easydict import EasyDict

import argparse
import json
from tqdm import tqdm

from models.get_model import get_model
from utils.config_parser import init_config
from utils.torch_utils import restore_model
from utils.flow_utils import flow_to_image, resize_flow, writeFlowKITTI
from datasets.flow_datasets import KITTIFlow
from transforms import sep_transforms

parser = argparse.ArgumentParser(description='create_submission',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model-folder", required=True, type=str, help='The model folder (that contains the configuration files and the model file)')
parser.add_argument("--output-dir", default=None, type=str, help="Output directory; default is test_flow under the folder of the model")
parser.add_argument("--trained-model", required=True, type=str, help="Trained model path in the model folder")
parser.add_argument("--set", required=True, choices=["training", "testing"], type=str, default="testing", help="Training set or testing set")


@torch.no_grad()
def create_kitti_submission(model, args):
    """ Create submission for the KITTI leaderboard """
    
    split = args.set
    
    input_transform = transforms.Compose([
        sep_transforms.Zoom(args.img_height, args.img_width),
        sep_transforms.OneHotSemantics(),
        sep_transforms.ArrayToTensor()
    ])  
    
    dataset_2012 = KITTIFlow(Path(args.root_kitti12) / split, Path(args.root_kitti12_sem) / split, name='kitti2012', input_transform=input_transform, test_mode=True)
    dataset_2015 = KITTIFlow(Path(args.root_kitti15) / split, Path(args.root_kitti15_sem) / split, name='kitti2015', input_transform=input_transform, test_mode=True)
        

    # start inference
    model.eval()
    for ds in [dataset_2012, dataset_2015]:
        ds_dir = args.output_dir / ds.name
        ds_dir.makedirs_p()
    
        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, pin_memory=True, shuffle=False)
        for i_step, data in tqdm(enumerate(data_loader)):
            def tensor2array(tensor):
                return tensor.detach().cpu().numpy().transpose([0, 2, 3, 1])  
    
            img1, img2 = data['img1'].cuda(), data['img2'].cuda()
            sem1, sem2 = data['sem1'].cuda(), data['sem2'].cuda()
    
            # compute output
            flows = model(img1, img2, sem1, sem2, with_bk=False)
            flow_pred = flows['flows_fw'][0]
            h, w = data['im_shape']
            h, w = h.item(), w.item()
            flow_pred_up = resize_flow(flow_pred, (h, w))
            
            filename = os.path.basename(data['img1_path'][0])
            output_file = ds_dir / 'flow' / filename
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            writeFlowKITTI(output_file, tensor2array(flow_pred_up)[0])                              
            
    print('Completed!')
    return      


@torch.no_grad()
def main():
    args = parser.parse_args()
    args.model_folder = Path(args.model_folder)
    
    if args.output_dir is None:
        args.output_dir = args.model_folder / args.set + '_flow_kitti'
    args.output_dir.makedirs_p()
    
    ## set up the model
    config_file = args.model_folder / 'config.json'
    model_file = args.model_folder / args.trained_model
    
    cfg = init_config(config_file)
        
    model = get_model(cfg.model).cuda()

    model = restore_model(model, model_file)
    model.eval()
    
    args.img_height, args.img_width = 256, 832
    
    args.root_kitti12 = "data/KITTI-2012/"
    args.root_kitti12_sem = "results/sdcnet/KITTI-2012/"  # semantic input path
    args.root_kitti15 = "data/KITTI-2015/"
    args.root_kitti15_sem = "results/sdcnet/KITTI-2015/"  # semantic input path
    
    create_kitti_submission(model, args)

    
if __name__ == '__main__':
    main()




