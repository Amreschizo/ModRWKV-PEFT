from world.model import RWKV
from world.world_encoder import WorldEncoder
from src.peft_loading import load_peft_model_for_multimodal

def WorldLoading(args):
    config = {
        'encoder_type': args.encoder_type,
        'encoder_path': args.encoder_path,
        'project_dim' : args.n_embd
        }
    modality = WorldEncoder(**config)
    
    model = RWKV(args, modality=modality)
    #model = RWKV(args)
    print(model)

    # Handle modality encoder freezing
    if 'moda' not in args.train_step:
        for param in model.modality.world_encoder.model.parameters():
            param.requires_grad = False
    if 'adapter' not in args.train_step:
        for param in model.modality.world_encoder.adapter.parameters():
            param.requires_grad = False
    
    # Note: Quantization should be applied AFTER weights are loaded
    # This is handled in world_train.py after load_state_dict
    
    # Handle RWKV backbone training configuration
    if 'rwkv' not in args.train_step:
        # Freeze RWKV completely
        for param in model.emb.parameters():
            param.requires_grad = False
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.ln_out.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = False
    elif args.peft != 'none':
        # Apply PEFT (LoRA) - this will freeze base RWKV and add trainable LoRA adapters
        # First freeze all RWKV parameters
        for param in model.emb.parameters():
            param.requires_grad = False
        for param in model.blocks.parameters():
            param.requires_grad = False
        for param in model.ln_out.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = False
        
        # Apply PEFT which will add trainable LoRA adapters
        model = load_peft_model_for_multimodal(model, args)
    else:
        # Full fine-tuning: all RWKV parameters are trainable (default behavior)
        pass
    
    return model