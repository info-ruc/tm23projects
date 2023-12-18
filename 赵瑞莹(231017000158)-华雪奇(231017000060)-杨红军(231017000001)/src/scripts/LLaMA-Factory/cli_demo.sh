CUDA_VISIBLE_DEVICES=6 python src/cli_demo.py \
    --model_name_or_path /home/ubuntu/LLaMA-Factory/HYX_Baichuan2_13B_v5 \
    --template bianque \
    --do_sample True \
    --repetition_penalty 1.05