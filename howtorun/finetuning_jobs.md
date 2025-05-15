
Replace model name with - what needs to be finetuned.
- --model_name Qwen/Qwen2.5-7B-Instruct 
- update the paths as required, --data_path, --output_dir, --deepspeed, --cache_dir, ~/logs


```bash
screen -S trt_finetune
screen -r trt_finetune

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetuning/sft_talker.py --model_name Qwen/Qwen2.5-7B-Instruct --bf16 --data_path ~/tr_data/train_mathdial.json --output_dir ~/model_checkpoints/trt_qwen_2point5_7b --cache_dir ~/cache --model_max_length 2048 --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --save_strategy "steps" --save_steps 50 --save_total_limit 5 --learning_rate 2e-5 --deepspeed ~/deepspeed/config.json 2>&1 | tee ~/logs/sft_trt_qwen_2point5_7b.log
```


