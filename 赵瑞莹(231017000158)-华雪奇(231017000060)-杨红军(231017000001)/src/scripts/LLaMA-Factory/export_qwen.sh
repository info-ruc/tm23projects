python src/export_model.py \
    --model_name_or_path /home/ubuntu/huggingface/Qwen-14B-Chat \
    --template chatml \
    --finetuning_type lora \
    --checkpoint_dir Qwen_for_match \
    --export_dir Qwen_for_match_merged