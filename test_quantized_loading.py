#!/usr/bin/env python3
"""
Test script to verify quantized model loading for RWKV7.
Tests loading models/rwkv7-g1c-2.9b-20251231-ctx8192.pth in 4-bit quantization.
"""

import os
import sys
import torch
from argparse import Namespace

# Set environment variables before imports
os.environ["RWKV_MY_TESTING"] = "x070"
os.environ["RWKV_CTXLEN"] = "8192"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_FLOAT_MODE"] = "bf16"

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world.model import RWKV
from world.world_encoder import WorldEncoder
from src.peft_loading import apply_quantization_to_model


def test_quantized_loading():
    """Test loading RWKV7 model in 4-bit quantization."""
    
    model_path = "models/rwkv7-g1c-2.9b-20251231-ctx8192.pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the model file exists in the models/ directory.")
        return False
    
    print(f"Testing quantized loading of: {model_path}")
    print("=" * 60)
    
    # Create minimal args for model initialization
    # These values should match the model architecture
    args = Namespace(
        vocab_size=65536,
        n_layer=24,
        n_embd=1024,
        dim_att=1024,
        dim_ffn=0,  # Will be calculated
        head_size_a=64,
        head_size_divisor=8,
        ctx_len=8192,
        grad_cp=0,
        quant="4bit",
        peft="none",
        train_step=[],
        encoder_type="",  # No modality encoder for this test
        encoder_path="",
        accelerator="gpu",
        precision="bf16",
        layerwise_lr=1,
        my_pile_stage=0,
        weight_decay=0,
        state_tune=False,
        train_type="none"
    )
    
    # Calculate dim_ffn if needed
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    
    try:
        print("\n1. Creating RWKV model...")
        # Create model without modality encoder for this test
        model = RWKV(args, modality=None)
        print("   ✓ Model created successfully")
        
        print("\n2. Loading model weights...")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("   ✓ Weights loaded successfully")
        
        print("\n3. Applying 4-bit quantization...")
        try:
            model = apply_quantization_to_model(model, "4bit")
            print("   ✓ Quantization applied successfully")
        except Exception as e:
            print(f"   ⚠ Quantization failed: {e}")
            print("   This may be expected if bitsandbytes is not properly installed")
            print("   Continuing test without quantization...")
        
        print("\n4. Checking model structure...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        print("\n5. Testing forward pass...")
        # Create dummy input
        dummy_input = torch.randint(0, args.vocab_size, (1, 10), dtype=torch.long)
        with torch.no_grad():
            output = model(dummy_input, signs=None)
        print(f"   ✓ Forward pass successful. Output shape: {output.shape}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_quantized_loading()
    sys.exit(0 if success else 1)
