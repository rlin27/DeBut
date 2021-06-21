# resnet-50: layer4.1.conv1: [512,2048*1]
# chain 1
CUDA_VISIBLE_DEVICES=1 python main.py \
--type_init ALS3 \
--sup 512 512 512 1024 1024 1024 1024 1024 1024 1024 1024 2048 2048 2048 2048 2048 2048 2048 \
--sub 2 2 256 2 4 128 2 2 64 2 2 32 2 2 16 2 4 8 2 2 4 2 2 2 2 2 1 \
--iter 20 \
--model resnet_50 \
--layer_name layer4[1].conv1 \
--log_path ./log_files_res/layer4_1_conv1_1 \
--gpu 1