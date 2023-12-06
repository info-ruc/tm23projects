python src/export_model.py \
    --model_name_or_path /home/ubuntu/huggingface/Baichuan2-13B-Base \
    --template bianque \
    --finetuning_type lora \
    --checkpoint_dir Baichuan2_save_sft_v4 \
    --output_dir HYX_Baichuan2_13B_v4_ \
    --bf16 True \
    --fp16 False