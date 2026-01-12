#!/bin/bash
# Stage 1: Train vision adapter only for RWKV-7
# This script trains only the adapter layer, keeping the RWKV LLM frozen

# Model configuration
load_model=models/rwkv7-g1c-2.9b-20251231-ctx8192.pth
proj_dir=out/vision-adapter-stage1-siglip

# Dataset configuration
data_file=datasets/LLaVA-CC3M-Pretrain-595K
data_type=visual  # Uses chat.jsonl format

# Encoder configuration (local SigLIP encoder)
encoder_path=encoders/siglip2-so400m-patch14-384
encoder_type=siglip

# Model architecture (for RWKV7-2.9B)
# These will be auto-detected from the model file, but you can override if needed
# Typical values for RWKV7-2.9B: n_layer=24, n_embd=1024 or n_layer=32, n_embd=2560
# Check your model file to confirm the exact architecture
n_layer=24
n_embd=1024

# Training hyperparameters (from ModRWKV paper Table 9 - Step1)
micro_bsz=256  # Batch size from paper
epoch_save=1
epoch_steps=10000  # Adjust based on dataset size
ctx_len=2048
lr_init=1e-3  # Learning rate from paper
lr_final=0
warmup_steps=100  # From paper
beta1=0.9
beta2=0.99
adam_eps=1e-8
lr_schedule=wsd  # From paper

# Hardware configuration
accelerator=gpu
devices=1  # Adjust based on your GPU setup
precision=bf16
strategy=deepspeed_stage_1
grad_cp=1

# Quantization (4-bit for LLM, even though it's frozen in Stage 1)
quant=4bit

# First, convert chat.json to chat.jsonl if needed
if [ ! -f "${data_file}/chat.jsonl" ]; then
    echo "Converting chat.json to chat.jsonl..."
    python3 convert_chat_json_to_jsonl.py "${data_file}/chat.json" "${data_file}/chat.jsonl"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert chat.json to chat.jsonl"
        exit 1
    fi
fi

# Extract images from zip if needed
# The dataset code expects images at: ${data_file}/${img_name}
# So images should be in the dataset root directory, not a subdirectory
if [ ! -f "${data_file}/GCC_train_002582585.jpg" ] && [ -f "${data_file}/images.zip" ]; then
    echo "Extracting images.zip to dataset root..."
    cd "${data_file}" && unzip -q -j images.zip "*.jpg" -d . 2>/dev/null || unzip -q images.zip && cd - || exit 1
    echo "Images extracted. Verifying..."
    if [ -f "${data_file}/GCC_train_002582585.jpg" ] || [ -d "${data_file}/images" ]; then
        echo "✓ Images ready"
    else
        echo "⚠ Warning: Image extraction may need manual verification"
    fi
fi

echo "Starting Stage 1 training: Vision adapter only"
echo "================================================"
echo "Model: $load_model"
echo "Encoder: $encoder_path ($encoder_type)"
echo "Dataset: $data_file ($data_type)"
echo "Output: $proj_dir"
echo "================================================"

HF_ENDPOINT="https://hf-mirror.com" python world_train.py \
--load_model $load_model \
--proj_dir $proj_dir \
--data_file $data_file \
--data_type $data_type \
--vocab_size 65536 \
--n_layer $n_layer \
--n_embd $n_embd \
--ctx_len $ctx_len \
--micro_bsz $micro_bsz \
--epoch_steps $epoch_steps \
--epoch_count 1 \
--epoch_begin 0 \
--epoch_save $epoch_save \
--lr_init $lr_init \
--lr_final $lr_final \
--warmup_steps $warmup_steps \
--beta1 $beta1 \
--beta2 $beta2 \
--adam_eps $adam_eps \
--lr_schedule $lr_schedule \
--accelerator $accelerator \
--devices $devices \
--precision $precision \
--strategy $strategy \
--grad_cp $grad_cp \
--encoder_path $encoder_path \
--encoder_type $encoder_type \
--my_testing "x070" \
--train_step adapter \
--quant $quant

echo ""
echo "Stage 1 training complete!"
echo "Checkpoint saved to: $proj_dir"
