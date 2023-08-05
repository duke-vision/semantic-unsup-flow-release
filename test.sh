#!/bin/bash

python3 test.py --model-folder=pretrained_models/ours_baseline --trained-model=model_ckpt.pth.tar --set=testing
python3 test.py --model-folder=pretrained_models/ours_baseline+enc --trained-model=model_ckpt.pth.tar --set=testing
python3 test.py --model-folder=pretrained_models/ours_baseline+enc+aug --trained-model=model_ckpt.pth.tar --set=testing

## use the following lines to generate a zip file for submission
# cd pretrained_models/ours_baseline+enc+aug/testing_flow_kitti/kitti2012/flow
# zip ../../flow2012.zip *

# cd pretrained_models/ours_baseline+enc+aug/testing_flow_kitti/kitti2015
# zip -r ../kitti2015.zip *
