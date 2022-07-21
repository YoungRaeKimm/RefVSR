#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0,1 python -B -m torch.distributed.launch --nproc_per_node=2 --master_port=9002 run.py \
                        --is_train \
                        --mode ERVSR \
                        --config config_ERVSR \
                        --data RealMCVSR \
                        --data_offset ../sdb
                        -b 4 \
                        -th 4 \
                        -dl \
                        -ss \
                        -dist \
                        --is_crop_valid \
