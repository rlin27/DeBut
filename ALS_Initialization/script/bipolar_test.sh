# A possible chain for matrix of size [512, 4608]
python chain_test.py \
--sup 512 1024 1024 2048 2048 4096 4096 4096 4096 4096 4096 4096 4096 4608 \
--sub 2 4 256 2 4 128 2 4 64 2 2 32 2 2 16 2 2 8 8 9 1 \
--log_path ./log_files/chain_test/mat_512_4608 \
