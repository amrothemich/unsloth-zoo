#!/usr/bin/env python3
"""
Minimal reproducer for GPT-OSS cache position overflow issue in GRPO training.
This reproduces the specific issue where cache_position jumps from ~63 to 1262+.
"""
import modal

# Persistent volume for state
state_volume = modal.Volume.from_name("cache-reproducer-state", create_if_missing=True)

# Image with exact Databricks package versions - install everything together for compatibility
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget", "curl"])
    .run_commands([
        # Install ALL packages in single command for dependency resolution
        # Let pip resolve compatible versions for conflicting packages
        """pip install --upgrade \
            'torch==2.6.0+cu124' \
            --index-url https://download.pytorch.org/whl/cu124 && \
        pip install --upgrade \
            transformers==4.56.2 \
            datasets==3.5.0 \
            accelerate==1.5.2 \
            trl==0.23.0 \
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
            'git+https://github.com/amrothemich/unsloth-zoo.git@c8f0afc#egg=unsloth-zoo'"""
    ])
)

app = modal.App("cache-reproducer")

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=3600,  # 60 minutes - much longer timeout
    memory=32768,
    volumes={"/state": state_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/state/cache_logs",
        "TORCH_USE_CUDA_DSA": "1",
        "UNSLOTH_ABORT_ON_CACHE_CORRUPTION": "1",  # Enable conservative abort mode
    }
)
def reproduce_cache_issue():
    """Minimal reproducer for cache position overflow"""
    import os
    import torch
    import json
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
    
    print("="*80)
    print("REPRODUCING CACHE POSITION OVERFLOW ISSUE")
    print("="*80)
    
    # Verify we have the latest version with the critical fix
    import unsloth_zoo
    print(f"📦 Using unsloth_zoo from: {unsloth_zoo.__file__}")
    
    # Check if our critical fix is present
    from unsloth_zoo.temporary_patches.cache_position_fix import patch_sliding_window_cache_creation
    import inspect
    source = inspect.getsource(patch_sliding_window_cache_creation)
    has_critical_fix = "CRITICAL FIX: Use sliding_window (128), NOT max_cache_len (3299)!" in source
    print(f"🔍 Critical window size fix present: {has_critical_fix}")
    if not has_critical_fix:
        raise RuntimeError("CRITICAL FIX NOT FOUND! Modal is using old cached version. Please force rebuild image.")
    
    # Create cache log directory
    os.makedirs("/state/cache_logs", exist_ok=True)
    
    # Load GPT-OSS-20B model with sliding window 128
    print("Loading GPT-OSS-20B model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=3400,  # Large to trigger cache overflow
        dtype=None,  # Let unsloth auto-detect proper dtype (revert to working config)
        load_in_4bit=True,
    )
    
    # For training, we need to prepare the model properly
    FastLanguageModel.for_training(model)
    
    # Add LoRA for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42
    )
    
    # Setup tokenizer for GRPO
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print(f"✅ Model loaded - Sliding window: {getattr(model.config, 'sliding_window', None)}")
    
    # Create dataset with long prompts to trigger cache overflow
    # Based on logs, need prompts that lead to positions around 63 with duplicates
    long_prompt = """<|system|>
You are a senior prior-authorization nurse who has reviewed thousands of clinical requests.
Your job is to produce a logical representation mapping clinical questions to our ontology.
<|user|>
Procedure requested: Medical Imaging Study - Complex Multi-Stage Analysis

Clinical Background:
The patient presents with chronic symptoms requiring comprehensive evaluation. Previous diagnostic attempts have been inconclusive, necessitating advanced imaging protocols with contrast enhancement and multi-planar reconstruction capabilities.

Patient History:
- Primary diagnosis considerations include complex musculoskeletal disorders
- Secondary conditions involve neurological components  
- Tertiary factors include vascular implications
- Previous interventions attempted: conservative management protocols
- Current symptom severity: moderate to severe persistent discomfort
- Duration of symptoms: extended timeline requiring immediate attention

Previous Questions and Clinical Assessments:
1. Question: What is the primary anatomical region of concern?
   Answer: Lumbar spine with radicular symptom patterns extending bilaterally
   Clinical Notes: MRI findings suggest possible disc herniation with nerve root compression

2. Question: Has conservative treatment been adequately attempted?
   Answer: Yes, comprehensive physical therapy regimen completed over 8-week period
   Clinical Notes: Patient compliance excellent, minimal improvement documented

3. Question: Are there any contraindications to contrast-enhanced imaging?
   Answer: No known allergies, renal function within normal limits per recent laboratory results
   Clinical Notes: eGFR >60, no history of adverse reactions to iodinated contrast media

Current Assessment Question:
Question: What specific imaging protocol is most appropriate for definitive diagnosis?
Answer: Multi-sequence MRI with gadolinium enhancement, including T1, T2, STIR, and diffusion-weighted sequences

Generate a comprehensive JSON logic representation for this complex clinical scenario.
Additional Context: """ + "detailed medical assessment data requiring thorough analysis " * 150

    print(f"Prompt length: ~{len(tokenizer.encode(long_prompt))} tokens")
    
    # Create minimal dataset
    data = []
    for i in range(4):  # Small dataset for quick reproduction
        prompt = long_prompt + f"\nCase ID: {i}\n<|assistant|>\n"
        data.append({
            "prompt": prompt,
            "answer": 0,
            "ground_truth_logic": json.dumps({"case": i})
        })
    
    train_dataset = Dataset.from_list(data)
    
    # Calculate prompt lengths
    tokens = tokenizer.encode(train_dataset[0]['prompt'], add_special_tokens=False)
    max_prompt_length = len(tokens) + 100
    max_completion_length = 3400 - max_prompt_length
    
    print(f"Dataset - prompt tokens: {len(tokens)}, max_completion: {max_completion_length}")
    
    # GRPO config designed to trigger cache overflow
    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=2,  # Must be multiple of num_generations  
        gradient_accumulation_steps=1,  # Minimal for quick reproduction
        num_generations=2,  # Multiple generations trigger cache corruption
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=10,  # Increase steps to give more chance for cache overflow
        save_steps=10,
        output_dir="/state/output",
        beta=0.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        remove_unused_columns=False,
        seed=42,
    )
    
    # Simple reward function
    def simple_reward(completions, **kwargs):
        return [1.0] * len(completions)
    
    print("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[simple_reward],
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("\n" + "="*80)
    print("STARTING TRAINING - MONITORING FOR CACHE OVERFLOW")
    print("Expected: positions jump from ~63 to 1262-1263")
    print("="*80)
    
    try:
        # First test: Run a single forward + backward pass to verify cache fix
        print("\n" + "="*60)
        print("🧪 PHASE 1: TESTING SINGLE FORWARD/BACKWARD PASS")
        print("="*60)
        
        # Get a single batch for testing
        test_sample = train_dataset[0]
        test_inputs = tokenizer(test_sample['prompt'], return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        # Move to device and ensure proper dtype
        device = next(model.parameters()).device
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        
        # Note: Model is already in bfloat16 from from_pretrained, don't cast bitsandbytes models
        print(f"📊 Model dtype: {next(model.parameters()).dtype}")
        
        print(f"📊 Test input shape: {test_inputs['input_ids'].shape}")
        print("🚀 Running forward pass...")
        
        # Ensure model is in training mode and use proper dtype
        model.train()
        
        # Test forward pass with try-catch for better error handling
        try:
            outputs = model(**test_inputs, labels=test_inputs['input_ids'])
            loss = outputs.loss
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            # Try without labels first to isolate the issue
            print("🔄 Trying forward pass without labels...")
            outputs = model(**test_inputs)
            print(f"✅ Forward pass without labels works! Output shape: {outputs.logits.shape}")
            
            # Now try with labels but ensure they're properly formatted
            labels = test_inputs['input_ids'].clone()
            outputs = model(**test_inputs, labels=labels)
            loss = outputs.loss
        
        print(f"✅ Forward pass successful! Loss: {loss.item():.4f}")
        print("🔄 Running backward pass...")
        
        loss.backward()
        print("✅ BACKWARD PASS SUCCESSFUL - Cache fix is working!")
        
        # Clear gradients for training
        model.zero_grad()
        
        # Now test evaluation mode
        print("\n" + "="*60)
        print("🧪 PHASE 2: TESTING EVALUATION MODE")
        print("="*60)
        
        model.eval()
        with torch.no_grad():
            eval_outputs = model(**test_inputs)
            print(f"✅ EVALUATION SUCCESSFUL! Output shape: {eval_outputs.logits.shape}")
        
        # Test generation to ensure cache works in generation
        print("🎯 Testing generation...")
        with torch.no_grad():
            generated = model.generate(
                **test_inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            print(f"✅ GENERATION SUCCESSFUL! Generated shape: {generated.shape}")
        
        model.train()  # Back to training mode
        
        # Now run full GRPO training
        print("\n" + "="*60)
        print("🚀 PHASE 3: FULL GRPO TRAINING")
        print("="*60)
        
        # Run training with progress monitoring
        import time
        import threading
        
        # Progress monitor that prints every 2 minutes
        def progress_monitor():
            start_time = time.time()
            while training_active:
                time.sleep(120)  # Wait 2 minutes
                if training_active:
                    elapsed = time.time() - start_time
                    print(f"🕐 Training still running... {elapsed/60:.1f} minutes elapsed")
                    print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
                    
                    # Check for any cache logs generated so far
                    if os.path.exists("/state/cache_logs"):
                        log_files = [f for f in os.listdir("/state/cache_logs") 
                                   if f.startswith("unsloth_cache")]
                        if log_files:
                            print(f"   📋 Cache logs generated: {len(log_files)} files")
        
        training_active = True
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        
        print("🚀 Starting GRPO training with monitoring...")
        trainer.train()
        training_active = False
        
        print("✅ GRPO TRAINING COMPLETED SUCCESSFULLY!")
        print("✅ CONFIRMED: Forward pass, backward pass, evaluation, and GRPO training all work!")
        
        return {
            "success": True, 
            "cache_issue_reproduced": False,
            "forward_pass_success": True,
            "backward_pass_success": True,
            "evaluation_success": True,
            "grpo_training_success": True
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}")
        
        # Check if it's the cache position error we expect
        cache_error = ("cache_position" in error_msg or 
                      "illegal memory access" in error_msg or
                      "index out of bounds" in error_msg)
        
        if cache_error:
            print("🎯 SUCCESSFULLY REPRODUCED CACHE POSITION OVERFLOW!")
            print("This is the expected error - now the fix needs to be applied")
            
            # Check cache logs
            cache_logs = []
            if os.path.exists("/state/cache_logs"):
                cache_logs = [f for f in os.listdir("/state/cache_logs") 
                             if f.startswith("unsloth_cache")]
                print(f"📋 Cache logs generated: {cache_logs}")
            
            return {
                "success": True,
                "cache_issue_reproduced": True,
                "error_message": error_msg,
                "cache_logs": cache_logs
            }
        else:
            print(f"❌ Unexpected error (not cache-related): {error_msg}")
            return {
                "success": False,
                "cache_issue_reproduced": False,
                "error_message": error_msg
            }

@app.function(
    image=image,
    volumes={"/state": state_volume}
)
def check_logs():
    """Check and return cache logs"""
    import os
    
    logs_info = {}
    if os.path.exists("/state/cache_logs"):
        log_files = [f for f in os.listdir("/state/cache_logs") 
                    if f.startswith("unsloth_cache")]
        
        for log_file in log_files:
            log_path = os.path.join("/state/cache_logs", log_file)
            try:
                with open(log_path, 'r') as f:
                    content = f.read()
                logs_info[log_file] = {
                    "size": len(content),
                    "lines": len(content.splitlines()),
                    "has_failures": "FAILURE" in content,
                    "has_illegal_access": "illegal memory access" in content,
                    "preview": content[:1000] if content else ""
                }
            except Exception as e:
                logs_info[log_file] = {"error": str(e)}
    
    return logs_info

@app.local_entrypoint()
def main():
    print("🚀 Running minimal cache overflow reproducer with 60min timeout...")
    import time
    
    # Start the reproducer
    print("Starting training function...")
    handle = reproduce_cache_issue.spawn()
    
    # Monitor progress every 2 minutes
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds initially
    
    while True:
        try:
            # Try to get the result with a short timeout
            result = handle.get(timeout=check_interval)
            # If we get here, the function completed
            break
        except TimeoutError:
            # Function is still running
            elapsed = time.time() - start_time
            print(f"\n⏱️  Training running for {elapsed/60:.1f} minutes...")
            
            # Check logs periodically
            try:
                logs_info = check_logs.remote()
                if logs_info:
                    print(f"📋 Cache logs detected: {len(logs_info)} files")
                    for log_file, info in logs_info.items():
                        if isinstance(info, dict) and info.get("has_failures"):
                            print(f"   ⚠️  {log_file}: {info['lines']} lines, HAS FAILURES!")
                        elif isinstance(info, dict):
                            print(f"   📄 {log_file}: {info['lines']} lines")
            except Exception as e:
                print(f"   (Could not check logs: {e})")
            
            # Increase check interval for longer runs
            if elapsed > 600:  # After 10 minutes, check every 2 minutes
                check_interval = 120
            
            continue  # Continue the while loop
    
    # Function completed, we have the result
    print(f"\n🎯 Final Result: {result}")
    
    # Get final logs
    try:
        final_logs = check_logs.remote()
        if final_logs:
            print(f"\n📋 Final cache logs analysis:")
            for log_file, info in final_logs.items():
                if isinstance(info, dict):
                    print(f"  {log_file}:")
                    print(f"    Lines: {info.get('lines', 0)}")
                    print(f"    Has failures: {info.get('has_failures', False)}")
                    print(f"    Has illegal access: {info.get('has_illegal_access', False)}")
                    if info.get('preview'):
                        print(f"    Preview: {info['preview'][:200]}...")
        
        if result.get("cache_issue_reproduced"):
            print("\n✅ CACHE ISSUE SUCCESSFULLY REPRODUCED!")
            print("The illegal memory access error was triggered as expected")
        elif result.get("success"):
            print("\n🤔 Training completed without cache issue")
            print("This could mean either:")
            print("  1. The fixes are working correctly (good!)")
            print("  2. We need different parameters to trigger the issue")
        else:
            print(f"\n❌ Unexpected error: {result.get('error_message', 'Unknown')}")
            
    except Exception as e:
        print(f"\n💥 Error during final analysis: {e}")
        # Still try to check logs
        try:
            final_logs = check_logs.remote()
            if final_logs:
                print(f"\n📋 Available logs despite error:")
                for log_file, info in final_logs.items():
                    print(f"  {log_file}: {info}")
        except:
            pass

if __name__ == "__main__":
    main()