#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python ./train.py \
--log_dir ./logs/vgg16 \
--data_dir ./data \
--r_shape_txt ./rshapes/vgg.txt \
--dataset CIFAR10 \
--debut_layers 10 11 12 14 15 16 \
--arch VGG_DeBut \
--use_pretrain \
--pretrained_file /mnt/nfsdisk/jier/debut_vgg/vgg_16_bn.pt \
--use_ALS \
--DeBut_init_dir /mnt/nfsdisk/jier/debut_vgg/mono_chain_als/ \
--batch_size 64 \
--epochs 150 \
--learning_rate 0.01 \
--lr_decay_step 50,100 \
--momentum 0.9 \
--weight_decay 5e-4 \
--gpu 0,1