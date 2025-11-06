#!/usr/bin/env python3
"""
Modal-based systematic validation of sliding window cache behavior.
Tests our fixes to ensure they don't introduce silent corruption.
"""
import modal
import json

# Image with our fixed libraries
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget", "curl"])
    .run_commands([
        """pip install --upgrade \
            'torch==2.6.0+cu124' \
            --index-url https://download.pytorch.org/whl/cu124 && \
        pip install --upgrade \
            transformers==4.56.2 \
            datasets==3.5.0 \
            accelerate==1.5.2 \
            peft==0.17.1 \
            bitsandbytes==0.48.2 \
            numpy==1.26.4 \
            safetensors==0.4.4 \
            'tokenizers>=0.22.0,<=0.23.0' \
            'huggingface-hub>=0.34.0' \
            packaging==25.0 \
            protobuf==6.33.0 \
            requests==2.32.3 \
            tqdm==4.67.1 \
            psutil==7.1.3 \
            'git+https://github.com/amrothemich/unsloth.git@fix-grpo#egg=unsloth[cu124-torch260]' \
            'git+https://github.com/amrothemich/unsloth-zoo.git@fix-grpo#egg=unsloth-zoo'"""
    ])
)

app = modal.App("cache-validation")

# Persistent volume for logs
logs_volume = modal.Volume.from_name("cache-validation-logs", create_if_missing=True)

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=1800,  # 30 minutes
    memory=32768,
    volumes={"/logs": logs_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/logs",
        "TORCH_USE_CUDA_DSA": "1",
    }
)
def test_1_simple_generation():
    """Test 1: Simple generation within window bounds (baseline - should work perfectly)"""
    import torch
    from unsloth import FastLanguageModel
    import unsloth_zoo  # Apply our patches
    
    print("🧪 TEST 1: Simple Generation (within 128 window)")
    print("="*60)
    
    try:
        # Load model with our fixes
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            max_seq_length=256,  # Small to stay within window
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
        # Check that our fix is applied
        sliding_window = getattr(model.config, 'sliding_window', None)
        print(f"📊 Model sliding window: {sliding_window}")
        
        # Test with SHORT prompt (well within 128 tokens)
        test_prompt = "The quick brown fox jumps"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")
        print(f"📊 Model dtype: {next(model.parameters()).dtype}")
        
        # Generate a few tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Stay well within window
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        print(f"✅ Generated: {generated_text}")
        print(f"✅ Total tokens: {outputs.sequences.shape[1]}")
        
        return {
            "test": "simple_generation",
            "status": "pass",
            "input_tokens": inputs['input_ids'].shape[1],
            "total_tokens": outputs.sequences.shape[1],
            "generated_text": generated_text,
            "sliding_window": sliding_window
        }
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test": "simple_generation", 
            "status": "fail", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.function(
    gpu="A100-40GB", 
    image=image,
    timeout=1800,
    memory=32768,
    volumes={"/logs": logs_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/logs",
        "TORCH_USE_CUDA_DSA": "1",
    }
)
def test_2_window_boundary():
    """Test 2: Generation that crosses the 128-token sliding window boundary"""
    import torch
    from unsloth import FastLanguageModel
    import unsloth_zoo
    
    print("🧪 TEST 2: Window Boundary Crossing")
    print("="*60)
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            max_seq_length=300,  # Allow crossing boundary
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
        # Create a prompt that's close to 128 tokens
        base_prompt = "This is a test prompt that will be repeated many times to reach close to 128 tokens. "
        repeated_prompt = base_prompt * 8  # Should be ~90+ tokens
        
        inputs = tokenizer(repeated_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        print(f"📊 Input tokens: {input_length}")
        print(f"📊 Target: Generate past position 128 to test window wrapping")
        
        # Generate enough tokens to cross the 128 boundary
        tokens_to_generate = max(50, 135 - input_length)
        print(f"📊 Generating {tokens_to_generate} tokens")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=tokens_to_generate,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        total_length = outputs.sequences.shape[1]
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        print(f"✅ Total tokens: {total_length}")
        print(f"✅ Crossed window boundary: {total_length > 128}")
        print(f"✅ Generated text (last 100 chars): ...{generated_text[-100:]}")
        
        return {
            "test": "window_boundary",
            "status": "pass",
            "input_tokens": input_length,
            "total_tokens": total_length,
            "crossed_boundary": total_length > 128,
            "tokens_generated": tokens_to_generate
        }
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test": "window_boundary", 
            "status": "fail", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.function(
    gpu="A100-40GB",
    image=image, 
    timeout=1800,
    memory=32768,
    volumes={"/logs": logs_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", 
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/logs",
        "TORCH_USE_CUDA_DSA": "1",
    }
)
def test_3_consistency_check():
    """Test 3: Check that same prompt gives same output (deterministic consistency)"""
    import torch
    from unsloth import FastLanguageModel
    import unsloth_zoo
    
    print("🧪 TEST 3: Consistency Check")
    print("="*60)
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            max_seq_length=256,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate same prompt twice
        results = []
        for i in range(2):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,  # Deterministic
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(generated_text)
            print(f"Run {i+1}: {generated_text}")
        
        consistent = results[0] == results[1]
        print(f"✅ Consistency: {consistent}")
        
        if not consistent:
            print(f"⚠️  Result 1: {results[0]}")
            print(f"⚠️  Result 2: {results[1]}")
        
        return {
            "test": "consistency",
            "status": "pass" if consistent else "fail",
            "consistent": consistent,
            "results": results
        }
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test": "consistency", 
            "status": "fail", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=1800, 
    memory=32768,
    volumes={"/logs": logs_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1", 
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/logs",
        "TORCH_USE_CUDA_DSA": "1",
    }
)
def test_4_forward_backward():
    """Test 4: Test forward and backward pass (training scenario)"""
    import torch
    from unsloth import FastLanguageModel
    import unsloth_zoo
    
    print("🧪 TEST 4: Forward/Backward Pass")
    print("="*60)
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            max_seq_length=256,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Prepare for training
        FastLanguageModel.for_training(model)
        
        # Add minimal LoRA for training
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,  # Smaller rank for faster testing
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42
        )
        
        test_text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(test_text, return_tensors="pt", max_length=50, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"📊 Input shape: {inputs['input_ids'].shape}")
        
        # Test forward pass
        model.train()
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        print(f"✅ Forward pass successful! Loss: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        print(f"✅ Gradients computed: {len(grad_norms)} parameters")
        print(f"✅ Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
        
        model.zero_grad()
        
        return {
            "test": "forward_backward",
            "status": "pass",
            "loss": loss.item(),
            "grad_parameters": len(grad_norms),
            "avg_grad_norm": sum(grad_norms)/len(grad_norms) if grad_norms else 0
        }
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test": "forward_backward",
            "status": "fail", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.function(
    image=image,
    volumes={"/logs": logs_volume}
)
def analyze_logs():
    """Analyze cache debug logs for warnings and repairs"""
    import os
    import re
    
    print("🧪 LOG ANALYSIS: Checking for cache repairs and warnings")
    print("="*60)
    
    log_analysis = {
        "cache_corruptions": 0,
        "position_wraps": 0,
        "accumulation_preventions": 0,
        "repairs": 0,
        "warnings": [],
        "errors": []
    }
    
    if os.path.exists("/logs"):
        log_files = [f for f in os.listdir("/logs") if f.endswith('.log')]
        print(f"📋 Found {len(log_files)} log files")
        
        patterns = {
            "corruption": r"🚨.*CORRUPTED",
            "wrap": r"🔧 WRAPPING cache_position",
            "prevention": r"🔧 PREVENTING ACCUMULATION", 
            "repair": r"🔧 REPAIRED",
        }
        
        for log_file in log_files:
            log_path = os.path.join("/logs", log_file)
            try:
                with open(log_path, 'r') as f:
                    content = f.read()
                
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    if pattern_name == "corruption":
                        log_analysis["cache_corruptions"] += len(matches)
                    elif pattern_name == "wrap":
                        log_analysis["position_wraps"] += len(matches)
                    elif pattern_name == "prevention": 
                        log_analysis["accumulation_preventions"] += len(matches)
                    elif pattern_name == "repair":
                        log_analysis["repairs"] += len(matches)
                
                # Extract specific warnings/errors
                lines = content.split('\n')
                for line in lines:
                    if '⚠️' in line or 'WARNING' in line:
                        log_analysis["warnings"].append(line.strip())
                    if '❌' in line or 'ERROR' in line:
                        log_analysis["errors"].append(line.strip())
                        
            except Exception as e:
                print(f"⚠️  Could not read {log_file}: {e}")
    
    print(f"📊 Cache corruptions detected: {log_analysis['cache_corruptions']}")
    print(f"📊 Position wraps applied: {log_analysis['position_wraps']}")
    print(f"📊 Accumulation preventions: {log_analysis['accumulation_preventions']}")
    print(f"📊 Cache repairs performed: {log_analysis['repairs']}")
    print(f"📊 Warnings: {len(log_analysis['warnings'])}")
    print(f"📊 Errors: {len(log_analysis['errors'])}")
    
    return log_analysis

@app.local_entrypoint()
def main():
    """Run complete validation suite"""
    print("🚀 SLIDING WINDOW CACHE VALIDATION SUITE")
    print("="*80)
    print("Testing our cache position overflow fixes for correctness")
    print("="*80)
    
    # Define test functions
    test_functions = [
        ("Simple Generation", test_1_simple_generation),
        ("Window Boundary", test_2_window_boundary), 
        ("Consistency Check", test_3_consistency_check),
        ("Forward/Backward", test_4_forward_backward)
    ]
    
    results = []
    
    # Run tests
    for test_name, test_func in test_functions:
        print(f"\n🔄 Running {test_name}...")
        try:
            result = test_func.remote()
            results.append(result)
            
            status_emoji = "✅" if result["status"] == "pass" else "❌"
            print(f"{status_emoji} {test_name}: {result['status']}")
            
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append({
                "test": test_name.lower().replace(" ", "_"),
                "status": "crash",
                "error": str(e)
            })
    
    # Analyze logs
    print(f"\n🔄 Analyzing cache debug logs...")
    log_analysis = analyze_logs.remote()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r["status"] == "pass")
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for result in results:
        status_emoji = "✅" if result["status"] == "pass" else "❌"
        test_name = result.get("test", "unknown")
        print(f"{status_emoji} {test_name}: {result['status']}")
        if result["status"] == "fail" and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Log analysis summary
    print(f"\n📋 Cache Log Analysis:")
    print(f"   Corruptions detected: {log_analysis.get('cache_corruptions', 0)}")
    print(f"   Repairs performed: {log_analysis.get('repairs', 0)}")
    print(f"   Position wraps: {log_analysis.get('position_wraps', 0)}")
    
    # Overall assessment
    corruption_count = log_analysis.get('cache_corruptions', 0)
    repair_count = log_analysis.get('repairs', 0)
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if passed == total and corruption_count == 0:
        print("🎉 EXCELLENT: All tests passed, no cache corruptions detected")
        print("✅ Fixes appear to work correctly and prevent corruption")
    elif passed == total and corruption_count > 0 and repair_count >= corruption_count:
        print("⚠️  GOOD: All tests passed, corruptions detected but repaired")
        print("✅ Fixes are working to catch and repair issues")
    elif passed >= total * 0.75:
        print("⚠️  ACCEPTABLE: Most tests passed, some issues detected")
        print("🔍 Review needed before production use")
    else:
        print("🚨 PROBLEMATIC: Significant issues detected")
        print("❌ Do not use in production - more fixes needed")
    
    # Save results
    final_results = {
        "test_results": results,
        "log_analysis": log_analysis,
        "summary": {
            "tests_passed": passed,
            "tests_total": total,
            "pass_rate": passed / total if total > 0 else 0,
            "corruptions_detected": corruption_count,
            "repairs_performed": repair_count
        }
    }
    
    print(f"\n💾 Validation complete - {passed}/{total} tests passed")
    
    return final_results

if __name__ == "__main__":
    main()