source activate lla
export PYTHONPATH=.

# WANDB_MODE=disabled \
WANDB_DISABLED=False \
CUDA_VISIBLE_DEVICES=4 \
accelerate launch --main_process_port $(shuf -i25000-30000 -n1) \
run_dpo.py \
config_lora.yaml \
--load_in_4bit=true

# accelerate launch --main_process_port 25002 --num_processes=1 \

# ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 25002 \
# --config_file recipes/accelerate_configs/multi_gpu.yaml \

# Full training with ZeRO-3 on 8 GPUs
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml

# # LoRA training on a single GPU
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml

# # QLoRA 4-bit training on a single GPU
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml --load_in_4bit=true

# # LoRA training with ZeRO-3 on two or more GPUs
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes={num_gpus} scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml