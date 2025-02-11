GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    train_mvdream14.py \
                                    --pretrained_model_name_or_path "lzq49/mvdream-sd21-diffusers" \
                                    --train_data_dir "/root/data/sb_exp/journeydb" \
                                    --resolution 512 \
                                    --use_ema \
                                    --validation_prompts "A blue poison-dart frog sitting on a water lily." "a DSLR photo of a group of dogs playing poker" "a squirrel dressed like Henry VIII king of England." "a car made out of sushi." "a DSLR photo of a pomeranian dog" "a DSLR photo of a baby dragon drinking boba" "a bichon frise wearing academic regalia" "a zoomed out DSLR photo of a tiger wearing sunglasses and a leather jacket, riding a motorcycle"\
                                    --validation_steps 500 \
                                    --train_batch_size 1 \
                                    --gradient_accumulation_steps 16 \
                                    --set_grads_to_none \
                                    --guidance_scale 4.5 \
                                    --learning_rate 1.e-06 \
                                    --learning_rate_lora 1.e-03 \
                                    --lr_scheduler "constant" --lr_warmup_steps 0 \
                                    --lora_rank 128 --lora_alpha 108 \
                                    --num_train_epochs 5 \
                                    --checkpointing_steps 2000 \
                                    --mixed_precision "fp16" \
                                    --output_dir exp_debug