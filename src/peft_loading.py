import os
import torch
import importlib
import json

from peft import get_peft_model, LoraConfig, BoneConfig, MissConfig, TaskType, AdaLoraConfig, PrefixTuningConfig
from peft import *

# Import quantization config if available
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    BitsAndBytesConfig = None


class RWKVConfig:
    def __init__(self, n_embd=2048, n_layer=24):
        self.model_type = "rwkv"
        self.tie_word_embeddings = False
        self.n_embd = n_embd
        self.n_layer = n_layer

    def get(self, key, default=None):
        return getattr(self, key, default)


def apply_quantization_to_model(model, quant_type):
    """
    Apply quantization to the RWKV model using BitsAndBytes.
    Only quantizes the RWKV backbone, not modality encoders.
    
    Note: BitsAndBytes quantization replaces Linear layers with quantized versions.
    For QLoRA, quantization is applied before PEFT adapters are added.
    """
    if not HAS_BITSANDBYTES:
        raise ImportError("bitsandbytes and transformers are required for quantization. Install with: pip install bitsandbytes transformers")
    
    if quant_type == "none":
        return model
    
    import bitsandbytes as bnb
    
    # Quantize linear layers in RWKV blocks
    # This quantizes: receptance, key, value, output layers in each block
    # We need to collect modules to replace first, then replace them
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Skip modality encoder and adapter layers - keep them in full precision
            if 'modality' in name or 'world_encoder' in name:
                continue
            
            # Skip embedding and output head for now (can be added if needed)
            if 'emb' in name or 'head' in name:
                continue
            
            modules_to_replace.append((name, module))
    
    # Replace modules
    for name, module in modules_to_replace:
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        # Get parent module
        if parent_name:
            parent_module = model
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)
        else:
            parent_module = model
        
        # Create quantized replacement
        if quant_type == "4bit":
            quantized_linear = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16
            )
            # Copy weights (will be quantized on first use)
            with torch.no_grad():
                if hasattr(quantized_linear.weight, 'data'):
                    quantized_linear.weight.data = module.weight.data.clone()
                if module.bias is not None and hasattr(quantized_linear, 'bias') and quantized_linear.bias is not None:
                    quantized_linear.bias.data = module.bias.data.clone()
        elif quant_type == "8bit":
            quantized_linear = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False
            )
            # Copy weights
            with torch.no_grad():
                if hasattr(quantized_linear.weight, 'data'):
                    quantized_linear.weight.data = module.weight.data.clone()
                if module.bias is not None and hasattr(quantized_linear, 'bias') and quantized_linear.bias is not None:
                    quantized_linear.bias.data = module.bias.data.clone()
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
        
        # Replace the module
        setattr(parent_module, child_name, quantized_linear)
    
    return model


def load_peft_model_for_multimodal(model, args):
    """
    Apply PEFT (LoRA) to the RWKV model in a multimodal context.
    This function is called from world_load.py after the model is created.
    
    Args:
        model: The RWKV model with modality encoder
        args: Training arguments
    
    Returns:
        model: Model with PEFT applied (if peft != 'none')
    """
    if args.peft == 'none':
        return model
    
    # Get the RWKV backbone (not the modality encoder)
    # The model structure is: RWKV(pl.LightningModule) containing:
    # - emb, blocks, ln_out, head (RWKV backbone)
    # - modality (WorldEncoder with vision/speech encoder)
    
    # We need to apply PEFT only to the RWKV backbone
    # Extract the RWKV model structure for PEFT
    rwkv_backbone = model
    
    # Set config for PEFT
    if not hasattr(rwkv_backbone, 'config'):
        rwkv_backbone.config = RWKVConfig(n_embd=args.n_embd, n_layer=args.n_layer)
    
    # Dynamic PEFT config loading
    peft_dict = {
        "lora": LoraConfig,
        "miss": MissConfig,
        "adalora": AdaLoraConfig,
        "prefix": PrefixTuningConfig,
    }
    
    if args.peft not in peft_dict:
        raise ValueError(f"Unsupported PEFT method: {args.peft}. Supported: {list(peft_dict.keys())}")
    
    ConfigClass = peft_dict[args.peft]
    
    # Parse peft_config JSON string
    if hasattr(args, 'peft_config') and args.peft_config:
        if isinstance(args.peft_config, str):
            peft_args = json.loads(args.peft_config)
        else:
            peft_args = args.peft_config
    else:
        # Default LoRA config
        peft_args = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.01}
    
    # Create PEFT config
    peft_config = ConfigClass(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["receptance", "key", "value", "output"],
        **peft_args
    )
    
    # Apply PEFT to the model
    # Note: get_peft_model will wrap the model and add LoRA adapters
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model
