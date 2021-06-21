# Approximate any given matrix
CUDA_VISIBLE_DEVICES=1 python main.py \
--type_init ALS3 \
--sup 512 512 512 1024 1024 1024 1024 1024 1024 1024 1024 2048 2048 2048 2048 2048 2048 2048 \
--sub 2 2 256 2 4 128 2 2 64 2 2 32 2 2 16 2 4 8 2 2 4 2 2 2 2 2 1 \
--iter 20 \
--F_path ./pth/matrix.pth \
--log_path ./log_files_res/matrix_approx \
--gpu 1