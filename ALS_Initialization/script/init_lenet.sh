# LeNet: fc1 [128, 400]
# Table chain
CUDA_VISIBLE_DEVICES=1 python main.py \
--type_init ALS3 \
--sup 128 256 256 512 512 512 512 256 256 400 \
--sub 2 4 64 1 2 64 2 2 32 2 1 16 16 25 1 \
--iter 10 \
--model lenet \
--layer_name fc \
--pth_path ./pth/lenet.pt \
--log_path ./log_files_lenet/fc1_table_chain \
--layer_type fc \
--gpu 1