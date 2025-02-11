CUDA_VISIBLE_DEVICES=0 python infer_mvdream.py \
                        --pretrained_model_name_or_path "lzq49/mvdream-sd21-diffusers" \
                        --prompt "An octopus and a giraffe having cheesecake." \
                        --prompt_list "/workspace/SwiftBrush/prompt_list_df.txt" \
                        --checkpoint_path "/workspace/SwiftBrush/exp_mvdream13_same_T_4view_%2/checkpoint-22000/unet_ema" \
                        --resolution 512 --batch_size 4 --four_view \
                        --seed 42 --output_dir result_mvdream13_same_T_4view_%2/dreamfusion_enhance/22K 

 # a zoomed out DSLR photo of a wizard raccoon casting a spell
                        # --prompt_list "/workspace/SwiftBrush/prompt_list_df.txt" \
   #                     --prompt_list "/root/data/sb_exp/journeydb.txt" \
   #                     --prompt_list "/workspace/SwiftBrush/coco_prompts.txt" \
