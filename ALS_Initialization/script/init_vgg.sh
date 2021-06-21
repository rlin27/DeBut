# vgg: conv 1 [64, 576]
# monotonic chain
CUDA_VISIBLE_DEVICES=1 python main.py \
--type_init ALS3 \
--sup 64 72 72 144 144 288 288 576 \
--sub 8 9 8 2 4 4 1 2 4 4 8 1 \
--iter 10 \
--model vgg \
--layer_name features.conv1 \
--pth_path ./pth/vgg_16_bn.pt \
--log_path ./log_files_vgg/vgg_conv1_1 \
--gpu 1