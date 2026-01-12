# Stage 1: Vision Adapter Training Command

This command trains only the vision adapter layer for RWKV-7, keeping the LLM frozen and loaded in 4-bit quantization.

## Prerequisites

1. **Convert dataset format**: The dataset has `chat.json` but the code expects `chat.jsonl`
   ```bash
   python3 convert_chat_json_to_jsonl.py datasets/LLaVA-CC3M-Pretrain-595K/chat.json datasets/LLaVA-CC3M-Pretrain-595K/chat.jsonl
   ```

2. **Extract images**: Extract images.zip so images are in the dataset root directory
   ```bash
   cd datasets/LLaVA-CC3M-Pretrain-595K
   unzip -q images.zip
   # If images extract to a subdirectory, move them to root:
   # mv images/*.jpg .  (if needed)
   cd ../..
   ```

3. **Verify model architecture**: Check your model's n_layer and n_embd values
   ```python
   import torch
   z = torch.load('models/rwkv7-g1c-2.9b-20251231-ctx8192.pth', map_location='cpu', weights_only=True)
   n_layer = max([int(k.split('.')[1]) for k in z.keys() if 'blocks.' in k]) + 1
   n_embd = z['emb.weight'].shape[1]
   print(f'n_layer={n_layer}, n_embd={n_embd}')
   ```

## Training Command

```bash
HF_ENDPOINT="https://hf-mirror.com" python world_train.py \
--load_model models/rwkv7-g1c-2.9b-20251231-ctx8192.pth \
--proj_dir out/vision-adapter-stage1-siglip \
--data_file datasets/LLaVA-CC3M-Pretrain-595K \
--data_type visual \
--vocab_size 65536 \
--n_layer 24 \
--n_embd 1024 \
--ctx_len 2048 \
--micro_bsz 256 \
--epoch_steps 10000 \
--epoch_count 1 \
--epoch_begin 0 \
--epoch_save 1 \
--lr_init 1e-3 \
--lr_final 0 \
--warmup_steps 100 \
--beta1 0.9 \
--beta2 0.99 \
--adam_eps 1e-8 \
--lr_schedule wsd \
--accelerator gpu \
--devices 1 \
--precision bf16 \
--strategy deepspeed_stage_1 \
--grad_cp 1 \
--encoder_path encoders/siglip2-so400m-patch14-384 \
--encoder_type siglip \
--my_testing "x070" \
--train_step adapter \
--quant 4bit
```

## Key Points

- **Stage 1**: Only trains the adapter (`--train_step adapter`)
- **LLM**: Loaded in 4-bit quantization (`--quant 4bit`) and frozen
- **Encoder**: Uses local SigLIP encoder from `encoders/siglip2-so400m-patch14-384`
- **Dataset**: LLaVA-CC3M-Pretrain-595K with `visual` data type
- **Hyperparameters**: Based on ModRWKV paper Table 9 (Step1)

## Adjustments

- **Batch size**: Adjust `--micro_bsz` based on your GPU memory (paper uses 256)
- **Devices**: Change `--devices` to use multiple GPUs
- **Model architecture**: Verify and adjust `--n_layer` and `--n_embd` to match your model
- **Epoch steps**: Adjust `--epoch_steps` based on dataset size (595K samples)

## Output

Checkpoints will be saved to: `out/vision-adapter-stage1-siglip/`
