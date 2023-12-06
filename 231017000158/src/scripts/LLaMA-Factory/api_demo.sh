CUDA_VISIBLE_DEVICES=7 python src/api_demo.py \
    --model_name_or_path /home/ubuntu/LLaMA-Factory/HYX_Baichuan2_13B_v5 \
    --template bianque \
    --do_sample False \
    --repetition_penalty 1.05
