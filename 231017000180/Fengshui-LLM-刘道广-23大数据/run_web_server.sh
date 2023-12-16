CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --model_name_or_path /xxx/LLaMA-Efficient-Tuning-main/model/Baichuan2-13B-Chat \
    --finetuning_type lora \
    --checkpoint_dir /xxx/LLaMA-Efficient-Tuning/saves/Baichuan2-13B-Chat/lora/2023-12-15-15-13-18 \
    --quantization_bit 4 \
    --template baichuan2
