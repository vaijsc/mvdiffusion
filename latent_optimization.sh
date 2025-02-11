GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                     latent_optimization.py \
                                    --pretrained_model_name_or_path "lzq49/mvdream-sd21-diffusers" \
                                    --train_data_dir "../env/data/sb_exp/journeydb" "../env/data/sb_exp/objaverse_prompt" \
                                    --resolution 512 \
                                    --use_ema \
                                    --validation_prompts "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" \
                                    --validation_steps 20 \
                                    --train_batch_size 1 \
                                    --gradient_accumulation_steps 16 \
                                    --set_grads_to_none \
                                    --guidance_scale 6.5 \
                                    --guidance_scale_sd 4.5 \
                                    --learning_rate 0.01\
                                    --learning_rate_lora 0.001\
                                    --lr_scheduler "constant" --lr_warmup_steps 0 \
                                    --lora_rank 128 --lora_alpha 108 \
                                    --num_train_epochs 5 \
                                    --checkpointing_steps 10000 \
                                    --output_dir exp_debug_gpu_$FIRST_GPU
