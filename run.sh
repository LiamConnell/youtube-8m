
python train.py --model_name=$1 --graph=simple_conv_w_res --num_files=5000000000 --tstep_size=1e-4
python infer.py --model_name=$1
