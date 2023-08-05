#!/bin/bash

## To reproduce Table 1 (benchmark test)
python3 train.py -c configs/kitti_baseline.json --n_gpu=2 --exp_folder iccv2023 --name ours_baseline
python3 train.py -c configs/kitti_baseline+enc.json --n_gpu=2 --exp_folder iccv2023 --name ours_enc
python3 train.py -c configs/kitti_baseline+enc+aug.json --n_gpu=2 --exp_folder iccv2023 --name ours_final

## To reproduce Table 2 (benchmark test vs other semantic optical flow)
python3 train.py -c configs/kitti_baseline+enc+aug.json --n_gpu=2 --exp_folder iccv2023 --name ours_final

## To reproduce Table 3 (ablation study)
python3 train.py -c configs/kitti_base.json --n_gpu=2 --exp_folder iccv2023 --name ours_no_change
python3 train.py -c configs/kitti_enc1.json --n_gpu=2 --exp_folder iccv2023 --name ours_enc1
python3 train.py -c configs/kitti_enc2.json --n_gpu=2 --exp_folder iccv2023 --name ours_enc2
python3 train.py -c configs/kitti_enc3.json --n_gpu=2 --exp_folder iccv2023 --name ours_enc3
python3 train.py -c configs/kitti_enc4.json --n_gpu=2 --exp_folder iccv2023 --name ours_enc4
python3 train.py -c configs/kitti_up_only.json --n_gpu=2 --exp_folder iccv2023 --name ours_up_only
python3 train.py -c configs/kitti_baseline.json --n_gpu=2 --exp_folder iccv2023 --name ours_baseline
python3 train.py -c configs/kitti_baseline+enc.json --n_gpu=2 --exp_folder iccv2023 --name ours_enc
python3 train.py -c configs/kitti_baseline+aug.json --n_gpu=2 --exp_folder iccv2023 --name ours_aug
python3 train.py -c configs/kitti_baseline+enc+aug.json --n_gpu=2 --exp_folder iccv2023 --name ours_final

## To reproduce Table 4 (ablation study of different aug options)
python3 train.py -c configs/kitti_ablate_aug1.json --n_gpu=2 --exp_folder iccv2023 --name ours_aug_start100k
python3 train.py -c configs/kitti_ablate_aug2.json --n_gpu=2 --exp_folder iccv2023 --name ours_aug_vehicles_only
python3 train.py -c configs/kitti_ablate_aug3.json --n_gpu=2 --exp_folder iccv2023 --name ours_aug_focus_new_occ

## To reproduce Table 6 (generalization ability)
python3 train.py -c configs/cityscapes_base.json --n_gpu=2 --exp_folder iccv2023 --name ours_city_no_change
python3 train.py -c configs/cityscapes_baseline.json --n_gpu=2 --exp_folder iccv2023 --name ours_city_baseline
python3 train.py -c configs/cityscapes_baseline+enc.json --n_gpu=2 --exp_folder iccv2023 --name ours_city_enc
python3 train.py -c configs/cityscapes_baseline+enc+aug.json --n_gpu=2 --exp_folder iccv2023 --name ours_city_final

## Enjoy!
