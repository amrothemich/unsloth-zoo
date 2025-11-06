#!/usr/bin/env python3
"""
Diagnostic script to understand the cache position overflow issue.
This will run in Databricks to show us exactly what's happening.
"""

def diagnose_cache_issue():
    """Run diagnostics to understand the cache corruption"""
    import torch
    from unsloth import FastLanguageModel
    import unsloth_zoo  # Apply patches
    
    print("🔍 CACHE DIAGNOSTIC - Understanding the Issue")
    print("="*60)
    
    # Load the same model that's causing issues
    MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    MAX_SEQ_LENGTH = 3400  # Same as GRPO training
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    print(f"\n📋 Model Config:")
    print(f"  sliding_window: {getattr(model.config, 'sliding_window', 'NOT FOUND')}")
    print(f"  max_position_embeddings: {getattr(model.config, 'max_position_embeddings', 'NOT FOUND')}")
    print(f"  use_cache: {getattr(model.config, 'use_cache', 'NOT FOUND')}")
    
    # Inspect cache layers if they exist
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"  num_layers: {len(model.model.layers)}")
        
        # Check first layer attention
        first_layer = model.model.layers[0]
        if hasattr(first_layer, 'self_attn'):
            print(f"\n🔍 First Layer Attention:")
            attn = first_layer.self_attn
            print(f"  layer_idx: {getattr(attn, 'layer_idx', 'NOT FOUND')}")
            print(f"  attention type: {type(attn)}")
    
    # Create a test input to see what cache gets created
    print(f"\n🧪 Testing Cache Creation:")
    test_input = "Test prompt for cache analysis"
    inputs = tokenizer(test_input, return_tensors="pt", max_length=100, truncation=True)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"  Input length: {inputs['input_ids'].shape[1]} tokens")
    print(f"  Device: {device}")
    
    # Hook to capture cache creation
    cache_info = {}
    
    def capture_cache_creation(module, args, kwargs):
        if 'cache_position' in kwargs:
            cache_pos = kwargs['cache_position']
            cache_info['cache_position_shape'] = cache_pos.shape if hasattr(cache_pos, 'shape') else type(cache_pos)
            cache_info['cache_position_values'] = cache_pos.tolist() if hasattr(cache_pos, 'tolist') else str(cache_pos)
        return None
    
    # Hook into SlidingWindowLayer if it exists
    try:
        from transformers.cache_utils import SlidingWindowLayer
        original_update = SlidingWindowLayer.update
        
        def diagnostic_update(self, key_states, value_states, cache_kwargs):
            print(f"\n🔧 SlidingWindowLayer.update called:")
            print(f"  self._actual_window_size: {getattr(self, '_actual_window_size', 'NOT SET')}")
            print(f"  key_states.shape: {key_states.shape}")
            print(f"  cache_kwargs keys: {list(cache_kwargs.keys())}")
            
            if 'cache_position' in cache_kwargs:
                cache_pos = cache_kwargs['cache_position']
                print(f"  cache_position type: {type(cache_pos)}")
                print(f"  cache_position shape: {getattr(cache_pos, 'shape', 'NO SHAPE')}")
                if hasattr(cache_pos, 'shape') and cache_pos.shape[0] <= 10:
                    print(f"  cache_position values: {cache_pos.tolist()}")
                elif hasattr(cache_pos, 'shape'):
                    print(f"  cache_position first 5: {cache_pos[:5].tolist()}")
                    print(f"  cache_position last 5: {cache_pos[-5:].tolist()}")
            
            if hasattr(self, 'keys') and self.keys is not None:
                print(f"  existing cache shape: {self.keys.shape}")
            else:
                print(f"  no existing cache")
            
            # Call original and see what happens
            try:
                result = original_update(self, key_states, value_states, cache_kwargs)
                print(f"  ✅ Update successful")
                return result
            except Exception as e:
                print(f"  ❌ Update failed: {e}")
                raise
        
        SlidingWindowLayer.update = diagnostic_update
        print("✅ Hooked SlidingWindowLayer.update for diagnostics")
        
    except ImportError:
        print("❌ Could not hook SlidingWindowLayer (not available)")
    
    # Try a simple forward pass
    print(f"\n🚀 Running Forward Pass:")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ Forward pass successful")
        print(f"  Output shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test generation if forward pass worked
    print(f"\n🎯 Testing Generation (where corruption typically occurs):")
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=50,  # Long enough to potentially trigger issue
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"✅ Generation successful")
        print(f"  Generated shape: {generated.shape}")
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"  Generated: {generated_text[:200]}...")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊 Cache Info Captured:")
    for k, v in cache_info.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    diagnose_cache_issue()