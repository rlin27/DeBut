#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py \
--log_dir ./logs/imagenet_resnet \
--data_dir [directory to training data] \
--r_shape_txt ./rshapes/resnet50.txt \
--arch resnet50 \
--use_ALS \
--DeBut_init_dir [directory of the saved ALS files] \
--pretrained_file [path to the pretrained checkpoint file] \
--batch_size 128 \
--epochs 100 \
--learning_rate 0.1 \
--momentum 0.9 \
--weight_decay 1e-4 \
--label_smooth 0.1 \
--gpu 0,1,2,3

