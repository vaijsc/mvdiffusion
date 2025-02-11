# accelerate launch train_mvdream.py \
#   --pretrained_model_name_or_path "lzq49/mvdream-sd21-diffusers" \
#   --train_data_dir "journeydb"\
#   --resolution 256 \
#   --use_ema \
#   --validation_prompts "A blue poison-dart frog sitting on a water lily." "A DSLR photo of an ice cream sundae." "a Marshall guitar amplifier with a logo and wheels on a gray metal case."\
#   --validation_steps 50 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 8 \
#   --set_grads_to_none \
#   --guidance_scale 4.5 \
#   --learning_rate 1.e-06 \
#   --learning_rate_lora 1.e-03 \
#   --lr_scheduler "constant" --lr_warmup_steps 0 \
#   --lora_rank 64 --lora_alpha 108 \
#   --num_train_epochs 5 \
#   --checkpointing_steps 10000 \
#   --output_dir swiftbrush-output_journeydb

accelerate launch train_swiftbrush.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
  --train_data_dir "prompt_list_df" \
  --resolution 512 \
  --use_ema \
  --validation_prompts "A blue poison-dart frog sitting on a water lily." "A DSLR photo of an ice cream sundae." "a squirrel dressed like Henry VIII king of England." "a car made out of sushi." \
  --validation_steps 50 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --set_grads_to_none \
  --guidance_scale 4.5 \
  --learning_rate 1.e-06 \
  --learning_rate_lora 1.e-03 \
  --lr_scheduler "constant" --lr_warmup_steps 0 \
  --lora_rank 64 --lora_alpha 108 \
  --num_train_epochs 10000 \
  --checkpointing_steps 10000
