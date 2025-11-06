#!/usr/bin/env python3
"""
Minimal reproducer for GPT-OSS cache position overflow issue in GRPO training.
This reproduces the specific issue where cache_position jumps from ~63 to 1262+.
"""
import modal

# Persistent volume for state
state_volume = modal.Volume.from_name("cache-reproducer-state", create_if_missing=True)

# Image with pinned packages
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget", "curl"])
    .pip_install(
        "torch==2.6.0+cu124",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install([
        "transformers==4.56.2",
        "datasets==3.5.0",
        "accelerate==1.5.2",
        "trl==0.23.0",
        "peft==0.17.1",
        "bitsandbytes==0.48.2",
        "numpy==1.26.4",
        "safetensors==0.4.4"
    ])
    .run_commands([
        "pip install --upgrade git+https://github.com/amrothemich/unsloth.git@fix-grpo#egg=unsloth[cu124-torch260]",
        "pip install --upgrade git+https://github.com/amrothemich/unsloth-zoo.git@fix-grpo"
    ])
)

app = modal.App("cache-reproducer")

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=1800,  # 30 minutes
    memory=32768,
    volumes={"/state": state_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/state/cache_logs",
        "TORCH_USE_CUDA_DSA": "1",
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
    
    # Create cache log directory
    os.makedirs("/state/cache_logs", exist_ok=True)
    
    # Load GPT-OSS-20B model with sliding window 128
    print("Loading GPT-OSS-20B model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=3400,  # Large to trigger cache overflow
        dtype=None,
        load_in_4bit=True,
    )
    
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
        max_steps=3,  # Just 3 steps to reproduce issue quickly
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
        # Run training - this should trigger the cache issue
        trainer.train()
        print("✅ Training completed - no cache issue reproduced")
        return {"success": True, "cache_issue_reproduced": False}
        
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

@app.local_entrypoint()
def main():
    print("🚀 Running minimal cache overflow reproducer...")
    result = reproduce_cache_issue.remote()
    print(f"\n🎯 Result: {result}")
    
    if result.get("cache_issue_reproduced"):
        print("\n✅ CACHE ISSUE SUCCESSFULLY REPRODUCED!")
        print("Next step: Apply fixes to unsloth-zoo patches")
    else:
        print("\n❌ Cache issue not reproduced - may need different parameters")

if __name__ == "__main__":
    main()