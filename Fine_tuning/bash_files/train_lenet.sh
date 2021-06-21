#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python ./train.py \
--log_dir ./logs/lenet \
--data_dir ./data \
--r_shape_txt ./rshapes/lenet.txt \
--dataset MNIST \
--debut_layers 0 1 2 \
--arch LeNet_DeBut \
--use_pretrain \
--pretrained_file [path to the pretrained checkpoint file] \
--use_ALS \
--DeBut_init_dir [directory of the saved ALS files] \
--batch_size 64 \
--epochs 150 \
--learning_rate 0.01 \
--lr_decay_step 50,100 \
--momentum 0.9 \
--weight_decay 5e-4 \
--gpu 2
