import time
import imageio
import numpy as np
import random
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow
from transforms import sep_transforms
from transforms.ar_transforms import oc_transforms

from utils.semantics_utils import read_semantics


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, root_sem, name='', input_transform=None, co_transform=None, ap_transform=None):
        self.root = Path(root)
        self.root_sem = Path(root_sem)
        self.name = name
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.samples = self.collect_samples()
        
        # for semantic occlusion augmentation. Temp hack: should be turned on manually when needed
        self.find_obj_mask = []
        
        #import IPython; IPython.embed()
        
    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        
        images, semantics = [], []
        for p in s['imgs']:
            image = imageio.imread(self.root / p).astype(np.float32) / 255.
            sem, _ = read_semantics(self.root_sem / p)
            sem = sem[..., None]
            
            images.append(image)
            semantics.append(sem)
        
        return images, semantics

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        data = {}
        
        images, semantics = self._load_sample(self.samples[idx])

        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            images, semantics, _ = self.co_transform(images, semantics, {})
        
        if self.input_transform is not None:
            images, semantics = self.input_transform((images, semantics))
            
        data['img1'], data['img2'] = images
        data['sem1'], data['sem2'] = semantics
          
        # prepare object mask for occlusion augmentation if needed
        placeholder = np.ones(data['img1'].shape[-2:], dtype=np.float32)[None] * np.nan
        for obj in self.find_obj_mask:
            if obj == 'car':  # we combine car, truck, bus, train
                obj_list = oc_transforms.semantic_connected_components(data['sem1'], class_indices=[13, 14, 15, 16], width_range=(50, 300), height_range=(50, 150))        
                if len(obj_list) > 0:
                    data[obj + '_mask'] = random.choice(obj_list)[None].astype(np.float32)
                else:
                    data[obj + '_mask'] = placeholder
                    
            elif obj == 'pole':  # we combine pole, traffic light, traffic sign
                obj_mask = oc_transforms.find_semantic_group(data['sem1'], class_indices=[5, 6, 7], win_width=200)
                if obj_mask is not None:
                    data[obj + '_mask'] = obj_mask[None]
                else:
                    data[obj + '_mask'] = placeholder
                
        if self.ap_transform is not None:
            data['img1_ph'], data['img2_ph'] = self.ap_transform([data['img1'].clone(), data['img2'].clone()])
        
        return data


class KITTIRawFile(ImgSeqDataset):
    def __init__(self, root, root_sem, sp_file, name='kitti-raw', ap_transform=None,
                 input_transform=None, co_transform=None):
        self.sp_file = sp_file
        super(KITTIRawFile, self).__init__(root, root_sem, name,
                                           input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                samples.append({'imgs': sp[0:2]})
                samples.append({'imgs': sp[2:4]})
            return samples


class KITTIFlowMV(ImgSeqDataset):
    """
    This dataset is used for unsupervised training only
    """

    def __init__(self, root, root_sem, name='', 
                 input_transform=None, co_transform=None, ap_transform=None, ):
        super(KITTIFlowMV, self).__init__(root, root_sem, name,
                                          input_transform=input_transform,
                                          co_transform=co_transform,
                                          ap_transform=ap_transform)

    def collect_samples(self):
        flow_occ_dir = 'flow_' + 'occ'
        assert (self.root / flow_occ_dir).isdir()

        img_l_dir, img_r_dir = 'image_2', 'image_3'
        assert (self.root / img_l_dir).isdir() and (self.root / img_r_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            for img_dir in [img_l_dir, img_r_dir]:
                img_list = (self.root / img_dir).files('*{}*.png'.format(root_filename))
                img_list.sort()

                for st in range(len(img_list) - 1):
                    seq = img_list[st:st+2]
                    sample = {}
                    sample['imgs'] = []
                    for i, file in enumerate(seq):
                        frame_id = int(file[-6:-4])
                        if 12 >= frame_id >= 9:
                            break
                        sample['imgs'].append(self.root.relpathto(file))
                    if len(sample['imgs']) == 2:
                        samples.append(sample)
        return samples


class KITTIFlow(ImgSeqDataset):
    """
    This dataset is used for validation/test ONLY, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, root_sem, name='', input_transform=None, test_mode=False):
        self.test_mode = test_mode
        super(KITTIFlow, self).__init__(root, root_sem, name, input_transform=input_transform)
        
    def __getitem__(self, idx):
        s = self.samples[idx]

        imgs = [s['img1'], s['img2']]
        
        images, semantics = [], []
        for p in imgs:
            image = imageio.imread(self.root / p).astype(np.float32) / 255.
            sem, _ = read_semantics(self.root_sem / p)
            sem = sem[..., None]
            
            images.append(image)
            semantics.append(sem)

        raw_size = images[0].shape[:2]
        
        if self.test_mode:
            data = {
                'im_shape': raw_size,
                'img1_path': self.root / s['img1'],
            }
            
        else:
            data = {
                'flow_occ': self.root / s['flow_occ'],
                'flow_noc': self.root / s['flow_noc'],
                'im_shape': raw_size,
                'img1_path': self.root / s['img1'],
            }

        if self.input_transform is not None:
            images, semantics = self.input_transform((images, semantics))
            
        data.update({'img1': images[0], 'img2': images[1]})
        data.update({'sem1': semantics[0], 'sem2': semantics[1]})
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        try:
            img_dir = 'image_2'   # for KITTI 2015
            assert (self.root / img_dir).isdir()
        except:
            img_dir = 'colored_0'   # for KITTI 2012
            assert (self.root / img_dir).isdir()
            
        if self.test_mode:
            img1s = sorted((self.root / img_dir).glob('*_10.png'))
            img2s = sorted((self.root / img_dir).glob('*_11.png'))
            
            samples = []
            for img1, img2 in zip(img1s, img2s):
                samples.append({'img1': img1.relpath(self.root), 'img2': img2.relpath(self.root)})
            
            return samples            
            
        flow_occ_dir = 'flow_' + 'occ'
        flow_noc_dir = 'flow_' + 'noc'
        assert (self.root / flow_occ_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            flow_occ_map = flow_occ_dir + '/' + flow_map
            flow_noc_map = flow_noc_dir + '/' + flow_map
            s = {'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map}

            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).isfile() and (self.root / img2).isfile()
            s.update({'img1': img1, 'img2': img2})
            samples.append(s)
        return samples


class Cityscapes(ImgSeqDataset):
    def __init__(self, root, root_sem, sp_file, name='cityscapes', ap_transform=None,
                 input_transform=None, co_transform=None):
        self.sp_file = sp_file
        super(Cityscapes, self).__init__(root, root_sem, name,
                                         input_transform=input_transform,
                                         co_transform=co_transform,
                                         ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                samples.append({'imgs': sp})
        
        return samples

