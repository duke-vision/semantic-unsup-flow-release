import copy
from torchvision import transforms

from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms import sep_transforms
from datasets.flow_datasets import KITTIRawFile, KITTIFlowMV, KITTIFlow, Cityscapes


def get_dataset(cfg):

    input_transform = transforms.Compose([
        sep_transforms.OneHotSemantics(),
        sep_transforms.ArrayToTensor()
    ])

    co_transform = get_co_transforms(aug_args=cfg.data_aug)
    

    if cfg.type == 'KITTI_Raw':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIRawFile(
            cfg.root_raw,
            cfg.root_raw_sem,
            cfg.train_file,
            name='kitti-raw',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, cfg.root_kitti15_sem, name='kitti2015', input_transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, cfg.root_kitti12_sem, name='kitti2012', input_transform=valid_input_transform)       
        train_sets = [train_set]
        train_sets_epoches = [-1]  # use this dataset till the end
        valid_sets = [valid_set_1, valid_set_2]
        
    elif cfg.type == 'KITTI_MV':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12
        root_sem = cfg.root_kitti15_sem if cfg.train_15 else cfg.root_kitti12_sem

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = KITTIFlowMV(
            root_flow,
            root_sem,
            name='kitti-mv',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, cfg.root_kitti15_sem, name='kitti2015', input_transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, cfg.root_kitti12_sem, name='kitti2012', input_transform=valid_input_transform)       
        train_sets = [train_set]
        train_sets_epoches = [-1]  # use this dataset till the end
        valid_sets = [valid_set_1, valid_set_2]
        
    elif cfg.type == 'KITTI_Raw+MV':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12
        root_sem = cfg.root_kitti15_sem if cfg.train_15 else cfg.root_kitti12_sem
        ds_name = "kitti2015-mv" if cfg.train_15 else "kitti2012-mv"
        
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set_1 = KITTIRawFile(
            cfg.root_raw,
            cfg.root_raw_sem,
            cfg.train_file,
            name='kitti-raw',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )
        train_set_2 = KITTIFlowMV(
            root_flow,
            root_sem,
            name=ds_name,
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )
        train_set = ConcatDataset([train_set_1, train_set_2])
        train_set.name = train_set_1.name + '&' + train_set_2.name

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, cfg.root_kitti15_sem, name='kitti2015', input_transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, cfg.root_kitti12_sem, name='kitti2012', input_transform=valid_input_transform)
        train_sets = [train_set]
        train_sets_epoches = [-1]  # use this dataset till the end
        valid_sets = [valid_set_1, valid_set_2]
        
    elif cfg.type == 'KITTI_Raw+MV_2stage':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))
        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
                            
        train_set_1 = KITTIRawFile(
            cfg.root_raw,
            cfg.root_raw_sem,
            cfg.train_file,
            name='kitti-raw',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )
        train_set_2_1 = KITTIFlowMV(
            cfg.root_kitti15,
            cfg.root_kitti15_sem,
            name='kitti2015-mv',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )
        train_set_2_2 = KITTIFlowMV(
            cfg.root_kitti12,
            cfg.root_kitti12_sem,
            name='kitti2012-mv',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )
        train_set_2 = ConcatDataset([train_set_2_1, train_set_2_2])
        train_set_2.name = 'kitti-mv'
      
        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, cfg.root_kitti15_sem, name='kitti2015', input_transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, cfg.root_kitti12_sem, name='kitti2012', input_transform=valid_input_transform)
        train_sets = [train_set_1, train_set_2]
        train_sets_epoches = [cfg.epoches_raw, cfg.epoches_mv]
        valid_sets = [valid_set_1, valid_set_2]

    elif cfg.type == 'Cityscapes':
        train_input_transform = copy.deepcopy(input_transform)
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

        ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
        train_set = Cityscapes(
            cfg.root_cityscapes,
            cfg.root_cityscapes_sem,
            cfg.train_file,
            name='cityscapes',
            input_transform=train_input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform
        )

        valid_input_transform = copy.deepcopy(input_transform)
        valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

        valid_set_1 = KITTIFlow(cfg.root_kitti15, cfg.root_kitti15_sem, name='kitti2015', input_transform=valid_input_transform)
        valid_set_2 = KITTIFlow(cfg.root_kitti12, cfg.root_kitti12_sem, name='kitti2012', input_transform=valid_input_transform)       
        train_sets = [train_set]
        train_sets_epoches = [-1]  # use this dataset till the end
        valid_sets = [valid_set_1, valid_set_2]
        
    else:
        raise NotImplementedError(cfg.type)
            
    return train_sets, valid_sets, train_sets_epoches