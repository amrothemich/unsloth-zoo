#!/usr/bin/env python3
"""
Modal script for debugging GRPO with GPT-OSS-20B.
Follows the pattern: SFT training -> Save -> Load -> GRPO training
This closely models the actual grpo_unsloth.py workflow.
"""
import modal
import json
import pickle

# Persistent volume for state between calls
state_volume = modal.Volume.from_name("grpo-gpt-oss-state", create_if_missing=True)

# Base image with pinned packages matching Databricks environment
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget", "curl"])
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
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
        # Install unsloth and unsloth-zoo together as requested - using working pattern from modal_grpo_interactive.py
        "pip install --upgrade git+https://github.com/amrothemich/unsloth.git@fix-grpo#egg=unsloth[cu124-torch260]",
        "pip install --upgrade git+https://github.com/amrothemich/unsloth-zoo.git@fix-grpo"
    ])
)

app = modal.App("grpo-gpt-oss-debug")

class State:
    """Simple state container"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.results = {}

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    memory=32768,
    volumes={"/state": state_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/state/cache_logs",
        "TORCH_USE_CUDA_DSA": "1",
        "PYTHONPATH": "/usr/local/lib/python3.12/site-packages"
    }
)
def step1_check_environment():
    """Step 1: Check the environment"""
    import sys
    import torch
    import subprocess
    import os
    
    print("="*80)
    print("STEP 1: CHECKING ENVIRONMENT")
    print("="*80)
    
    # Create cache log directory
    os.makedirs("/state/cache_logs", exist_ok=True)
    
    result = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        result.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        })
        
        # nvidia-smi output
        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True
            )
            print("\nNVIDIA-SMI Output:")
            print(nvidia_smi.stdout[:500])  # First 500 chars
        except:
            pass
    
    # Print results
    print("\nEnvironment Details:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Save result
    with open("/state/step1_result.json", "w") as f:
        json.dump(result, f)
    
    return result

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    memory=32768,
    volumes={"/state": state_volume}
)
def step2_load_base_model():
    """Step 2: Load base GPT-OSS-20B model"""
    import unsloth  # Import unsloth first to trigger patches
    import unsloth_zoo  # Import to trigger our patches
    import torch
    import gc
    import os
    import pickle
    
    print("="*80)
    print("STEP 2: LOADING BASE GPT-OSS-20B MODEL")
    print("="*80)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['UNSLOTH_ENABLE_LOGGING'] = '1'
    
    print("\nMemory before loading:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    try:
        from unsloth import FastLanguageModel
        
        # Using the correct unsloth GPT-OSS model that triggers the cache position overflow
        MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit" 
        MAX_SEQ_LENGTH = 2048  # Start with 2048 for initial loading, will increase for GRPO
        
        print(f"\nLoading model: {MODEL_NAME}")
        print(f"Max sequence length: {MAX_SEQ_LENGTH}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # None for auto-detection (BF16 for Ampere+ GPUs)
            load_in_4bit=True,
            # Note: token would be needed for HF access, but unsloth models should be public
        )
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Right padding for SFT
        
        print("\nMemory after loading:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"\nModel type: {type(model)}")
        print(f"Model config sliding_window: {getattr(model.config, 'sliding_window', 'Not found')}")
        
        # Don't use pickle - we'll save via trainer.save_model() after SFT
        # Just store basic info for now
        print("✅ Base model loaded successfully - will add LoRA and save via SFT trainer")
        
        result = {
            "success": True,
            "model_name": MODEL_NAME,
            "model_type": str(type(model)),
            "sliding_window": getattr(model.config, 'sliding_window', None),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
        }
        
        with open("/state/step2_result.json", "w") as f:
            json.dump(result, f)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=1200,  # Longer for SFT training
    memory=32768,
    volumes={"/state": state_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/state/cache_logs",
    }
)
def step3_sft_training():
    """Step 3: Do SFT training (similar to grpo_unsloth.py initial training)"""
    import unsloth  # Import unsloth first
    import unsloth_zoo
    import torch
    import json as json_module
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    print("="*80)
    print("STEP 3: SFT TRAINING")
    print("="*80)
    
    # Load the base model fresh for SFT (since we can't pickle the model)
    try:
        from unsloth import FastLanguageModel
        
        MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit" 
        MAX_SEQ_LENGTH = 2048  # Use 2048 for SFT training
        
        print(f"Loading base model for SFT: {MODEL_NAME}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # None for auto-detection (BF16 for Ampere+ GPUs)
            load_in_4bit=True,
        )
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Right padding for SFT
        
    except Exception as e:
        return {"success": False, "error": f"Could not load base model: {e}"}
    
    print("\nMemory before SFT:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    try:
        from unsloth import FastLanguageModel
        
        LORA_R = 128  # Match your SFT setup exactly  
        MAX_SEQ_LENGTH = 2048  # Use 2048 for SFT training
        
        print(f"\nAdding LoRA with rank={LORA_R}")
        
        # Add LoRA adapters for SFT - match your exact configuration
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,  # 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=128,  # Best to choose alpha = rank or rank*2
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",     # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # "unsloth" uses 30% less VRAM
            random_state=42
        )
        
        print("\nMemory after LoRA:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Create SFT dataset mimicking grpo_unsloth.py style
        print("\nCreating SFT dataset...")
        sft_data = []
        
        # Create prompts similar to the medical logic extraction task
        for i in range(20):  # Small dataset for testing
            prompt = f"""<|system|>
You are a system. Generate JSON output.
<|user|>
Extract logic for condition {i}: Patient has symptoms including fever and headache. History of diabetes.
<|assistant|>
"""
            # Simple completion that should generate valid JSON
            completion = json_module.dumps({
                "node_type": "Concept",
                "name": f"Condition_{i}",
                "status": "true",
                "body_parts": ["head"],
                "symptoms": ["fever", "headache"]
            }, indent=2)
            
            sft_data.append({
                "text": prompt + completion + tokenizer.eos_token
            })
        
        sft_dataset = Dataset.from_list(sft_data)
        
        # SFT Training
        print("\nSetting up SFT trainer...")
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,  # Short SFT for testing
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="/state/sft_output",
            remove_unused_columns=False,
        )
        
        sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=sft_dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )
        
        print("\nRunning SFT training...")
        print("🔍 WATCHING FOR CACHE ISSUES DURING FORWARD PASSES...")
        sft_result = sft_trainer.train()
        
        print("\n✅ SFT training completed!")
        print("\nMemory after SFT:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Save the SFT model to disk (the proper way)
        print("\nSaving SFT model to disk...")
        sft_trainer.save_model("/state/sft_model")
        tokenizer.save_pretrained("/state/sft_model")
        
        # Check for cache debug logs
        import os
        log_files = []
        if os.path.exists("/state/cache_logs"):
            log_files = os.listdir("/state/cache_logs")
            if log_files:
                print(f"\n📋 Cache debug logs generated during SFT: {log_files}")
        
        result = {
            "success": True,
            "lora_r": LORA_R,
            "sft_steps": sft_result.global_step,
            "sft_loss": sft_result.metrics.get("train_loss", None),
            "model_saved": "/state/sft_model",
            "cache_logs": log_files,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
        }
        
        with open("/state/step3_result.json", "w") as f:
            json_module.dump(result, f)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    memory=32768,
    volumes={"/state": state_volume}
)
def step4_load_sft_model():
    """Step 4: Load saved SFT model from disk for GRPO"""
    import unsloth  # Import unsloth first
    import unsloth_zoo
    import torch
    import pickle
    import gc
    
    print("="*80)
    print("STEP 4: LOADING SAVED SFT MODEL FROM DISK")
    print("="*80)
    
    # Clear GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\nMemory before loading SFT model:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    try:
        from unsloth import FastLanguageModel
        
        MAX_SEQ_LENGTH = 3400  # Matching grpo_unsloth.py
        
        print(f"\nLoading SFT model from /state/sft_model...")
        print(f"Max sequence length for GRPO: {MAX_SEQ_LENGTH}")
        
        # Load the saved SFT model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/state/sft_model",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,  # Using bfloat16 for GRPO like grpo_unsloth.py
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth"
        )
        
        # Set up tokenizer for GRPO (left padding)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Left padding for GRPO
        
        print("\nMemory after loading SFT model:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Save new state
        state = State()
        state.model = model
        state.tokenizer = tokenizer
        
        with open("/state/model_state.pkl", "wb") as f:
            pickle.dump(state, f)
        
        result = {
            "success": True,
            "sft_model_loaded": True,
            "model_type": str(type(model)),
            "max_seq_length": MAX_SEQ_LENGTH,
            "sliding_window": getattr(model.config, 'sliding_window', None),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
        }
        
        with open("/state/step4_result.json", "w") as f:
            import json
            json.dump(result, f)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=600,
    memory=32768,
    volumes={"/state": state_volume}
)
def step5_create_grpo_trainer():
    """Step 5: Create GRPO trainer with dataset similar to grpo_unsloth.py"""
    import unsloth  # Import unsloth first
    import unsloth_zoo
    import torch
    import json as json_module
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    
    print("="*80)
    print("STEP 5: CREATING GRPO TRAINER")
    print("="*80)
    
    # Load the saved SFT model from disk (as saved in step4)
    try:
        from unsloth import FastLanguageModel
        
        MAX_SEQ_LENGTH = 3400  # Increase to 3400 for GRPO (this triggers the issue)
        
        print(f"Loading SFT model from /state/sft_model...")
        print(f"Max sequence length for GRPO: {MAX_SEQ_LENGTH}")
        
        # Load the saved SFT model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/state/sft_model",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # Use bfloat16 for GRPO like grpo_unsloth.py
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth"
        )
        
        # Set up tokenizer for GRPO (left padding)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Left padding for GRPO
        
        print("✅ SFT model loaded for GRPO")
        
    except Exception as e:
        return {"success": False, "error": f"Could not load SFT model: {e}"}
    
    print("\nMemory before trainer:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    try:
        # Constants tuned to reproduce cache position overflow around position 63
        MAX_SEQ_LENGTH = 3400
        MIN_COMPLETION_TOKENS = 1500
        BUFFER_TOKENS = 100
        
        # Create dataset mimicking grpo_unsloth.py with specific prompt patterns that trigger the issue
        print("\nCreating GRPO dataset...")
        
        # Generate prompts that are likely to trigger cache position overflow
        # Based on logs, we need prompts that lead to positions around 63 with duplicates
        base_prompt = """<|system|>
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
Additional Context: """ + "detailed medical assessment data requiring thorough analysis " * 150  # Targeted length
        
        data = []
        for i in range(10):
            prompt = base_prompt + f"\nCase ID: {i}\n<|assistant|>\n"
            
            # Ground truth for scoring
            ground_truth = json_module.dumps({
                "node_type": "Concept",
                "name": "Conservative_Treatment",
                "status": "true",
                "duration": {"time_quantity": 6, "unit": "week"},
                "treatment_type": ["physical_therapy"]
            })
            
            data.append({
                "prompt": prompt,
                "answer": 0,  # Not used in RL
                "ground_truth_logic": ground_truth
            })
        
        train_dataset = Dataset.from_list(data)
        val_dataset = Dataset.from_list(data[:2])
        
        # Calculate prompt lengths
        tokens = tokenizer.encode(train_dataset[0]['prompt'], add_special_tokens=False)
        max_prompt_length = len(tokens) + BUFFER_TOKENS
        max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
        
        print(f"\nDataset statistics:")
        print(f"  First prompt tokens: {len(tokens)}")
        print(f"  Max prompt length: {max_prompt_length}")
        print(f"  Max completion length: {max_completion_length}")
        
        if max_completion_length < MIN_COMPLETION_TOKENS:
            print(f"\nWARNING: Max completion length ({max_completion_length}) is less than")
            print(f"recommended minimum ({MIN_COMPLETION_TOKENS} tokens).")
        
        # Create trainer config tuned to reproduce cache position overflow
        training_args = GRPOConfig(
            temperature=1.0,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=2,  # Must be multiple of num_generations
            gradient_accumulation_steps=2,  # Reduce to maintain same effective batch size
            num_generations=2,  # Multiple generations can trigger cache corruption
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=100,  # More steps to increase chance of reproducing issue
            save_steps=50,
            output_dir="/state/grpo_output",
            beta=0.0,  # DPO beta=0 means pure GRPO
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            bf16_full_eval=True,
            eval_strategy="steps",
            eval_steps=50,
            per_device_eval_batch_size=2,  # Must match num_generations
            eval_accumulation_steps=1,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            # Additional settings that may trigger cache issues
            report_to=None,  # Disable wandb/tensorboard
            push_to_hub=False,
            hub_model_id=None,
            seed=42,
        )
        
        # Simple reward functions
        def valid_json_reward(completions, **kwargs):
            """Check if response contains valid JSON"""
            scores = []
            for completion in completions:
                try:
                    # Try to extract JSON
                    if "{" in completion and "}" in completion:
                        start = completion.find("{")
                        end = completion.rfind("}") + 1
                        json_module.loads(completion[start:end])
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                except:
                    scores.append(0.0)
            return scores
        
        def simple_scoring_reward(completions, prompts=None, outputs=None, **kwargs):
            """Simple reward based on JSON validity and basic structure"""
            scores = []
            batch = kwargs.get('batch', {})
            ground_truths = batch.get('ground_truth_logic', [])
            
            for idx, completion in enumerate(completions):
                try:
                    # Extract JSON
                    if "{" in completion and "}" in completion:
                        start = completion.find("{")
                        end = completion.rfind("}") + 1
                        parsed = json_module.loads(completion[start:end])
                        
                        # Basic scoring - check if it has expected fields
                        score = 0.0
                        if parsed.get("node_type") == "Concept":
                            score += 0.5
                        if "name" in parsed:
                            score += 0.3
                        if "status" in parsed:
                            score += 0.2
                        
                        scores.append(score * 10.0)  # Scale up
                    else:
                        scores.append(0.0)
                except:
                    scores.append(0.0)
            
            return scores
        
        print("\nCreating trainer...")
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[valid_json_reward, simple_scoring_reward],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Trainer created successfully - no need to save state
        
        print("\nMemory after trainer:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        result = {
            "success": True,
            "prompt_tokens": len(tokens),
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "sliding_window": getattr(model.config, 'sliding_window', None),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
        }
        
        with open("/state/step5_result.json", "w") as f:
            json_module.dump(result, f)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.function(
    gpu="A100-40GB",
    image=image,
    timeout=3600,  # 60 minutes for thorough debugging
    memory=32768,
    volumes={"/state": state_volume},
    env={
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_ENABLE_LOGGING": "1",
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/state/cache_logs",
        "TORCH_USE_CUDA_DSA": "1",
        "PYTHONPATH": "/usr/local/lib/python3.12/site-packages"
    }
)
def step6_run_grpo_training():
    """Step 6: Run GRPO training and monitor for cache issues"""
    import unsloth  # Import unsloth first
    import unsloth_zoo
    import torch
    import json
    import os
    
    print("="*80)
    print("STEP 6: RUNNING GRPO TRAINING")
    print("="*80)
    
    # Recreate the GRPO trainer fresh (different from SFT trainer)
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
        import json as json_module
        
        MAX_SEQ_LENGTH = 3400  # Use 3400 for GRPO (this triggers the cache issue)
        
        print(f"Loading SFT model for GRPO training...")
        
        # Load the saved SFT model from step3
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/state/sft_model",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth"
        )
        
        # Set up tokenizer for GRPO (left padding)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        print("✅ SFT model loaded for GRPO training")
        print(f"✅ Sliding window: {getattr(model.config, 'sliding_window', None)}")
        print(f"✅ Max seq length: {MAX_SEQ_LENGTH}")
        
        # Recreate the GRPO dataset and trainer (same as step5)
        MIN_COMPLETION_TOKENS = 1500
        BUFFER_TOKENS = 100
        
        print("\nCreating GRPO dataset...")
        
        # Generate prompts that are likely to trigger cache position overflow
        base_prompt = """<|system|>
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
Additional Context: """ + "detailed medical assessment data requiring thorough analysis " * 150  # Targeted length
        
        data = []
        for i in range(10):
            prompt = base_prompt + f"\nCase ID: {i}\n<|assistant|>\n"
            
            # Ground truth for scoring
            ground_truth = json_module.dumps({
                "node_type": "Concept",
                "name": "Conservative_Treatment",
                "status": "true",
                "duration": {"time_quantity": 6, "unit": "week"},
                "treatment_type": ["physical_therapy"]
            })
            
            data.append({
                "prompt": prompt,
                "answer": 0,  # Not used in RL
                "ground_truth_logic": ground_truth
            })
        
        train_dataset = Dataset.from_list(data)
        val_dataset = Dataset.from_list(data[:2])
        
        # Calculate prompt lengths
        tokens = tokenizer.encode(train_dataset[0]['prompt'], add_special_tokens=False)
        max_prompt_length = len(tokens) + BUFFER_TOKENS
        max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
        
        print(f"\nDataset statistics:")
        print(f"  First prompt tokens: {len(tokens)}")
        print(f"  Max prompt length: {max_prompt_length}")
        print(f"  Max completion length: {max_completion_length}")
        
        if max_completion_length < MIN_COMPLETION_TOKENS:
            print(f"\nWARNING: Max completion length ({max_completion_length}) is less than")
            print(f"recommended minimum ({MIN_COMPLETION_TOKENS} tokens).")
        
        # Create GRPO trainer config
        training_args = GRPOConfig(
            temperature=1.0,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=2,  # Must be multiple of num_generations
            gradient_accumulation_steps=2,  # Reduce to maintain same effective batch size
            num_generations=2,  # Multiple generations can trigger cache corruption
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=100,  # More steps to increase chance of reproducing issue
            save_steps=50,
            output_dir="/state/grpo_output",
            beta=0.0,  # DPO beta=0 means pure GRPO
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            bf16_full_eval=True,
            eval_strategy="steps",
            eval_steps=50,
            per_device_eval_batch_size=2,  # Must match num_generations
            eval_accumulation_steps=1,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=None,
            push_to_hub=False,
            hub_model_id=None,
            seed=42,
        )
        
        # Simple reward functions
        def valid_json_reward(completions, **kwargs):
            """Check if response contains valid JSON"""
            scores = []
            for completion in completions:
                try:
                    # Try to extract JSON
                    if "{" in completion and "}" in completion:
                        start = completion.find("{")
                        end = completion.rfind("}") + 1
                        json_module.loads(completion[start:end])
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                except:
                    scores.append(0.0)
            return scores
        
        def simple_scoring_reward(completions, prompts=None, outputs=None, **kwargs):
            """Simple reward based on JSON validity and basic structure"""
            scores = []
            batch = kwargs.get('batch', {})
            ground_truths = batch.get('ground_truth_logic', [])
            
            for idx, completion in enumerate(completions):
                try:
                    # Extract JSON
                    if "{" in completion and "}" in completion:
                        start = completion.find("{")
                        end = completion.rfind("}") + 1
                        parsed = json_module.loads(completion[start:end])
                        
                        # Basic scoring - check if it has expected fields
                        score = 0.0
                        if parsed.get("node_type") == "Concept":
                            score += 0.5
                        if "name" in parsed:
                            score += 0.3
                        if "status" in parsed:
                            score += 0.2
                        
                        scores.append(score * 10.0)  # Scale up
                    else:
                        scores.append(0.0)
                except:
                    scores.append(0.0)
            
            return scores
        
        print("\nCreating GRPO trainer...")
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[valid_json_reward, simple_scoring_reward],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
    except Exception as e:
        return {"success": False, "error": f"Could not reload model for GRPO: {e}"}
    
    print("\nMemory before training:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Clear any existing cache logs
    if os.path.exists("/state/cache_logs"):
        for f in os.listdir("/state/cache_logs"):
            if f.startswith("unsloth_cache"):
                os.remove(os.path.join("/state/cache_logs", f))
    
    try:
        print("\n🚀 Starting GRPO training...")
        print("🔍 MONITORING FOR CACHE POSITION ISSUES...")
        print("🔍 Sliding window size:", getattr(model.config, 'sliding_window', 'Not set'))
        print("🔍 Expected issue: cache positions jumping from ~63 to 1262-1263")
        print("🔍 Watching for illegal memory access and position overflow patterns")
        print("-" * 80)
        
        # Add extra monitoring for cache position behavior
        import time
        start_time = time.time()
        
        # Run training with detailed monitoring
        print("Starting trainer.train() - this should trigger cache position overflow...")
        train_result = trainer.train()
        
        print("-" * 40)
        print("\n✅ Training completed successfully!")
        
        print("\nMemory after training:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # Check cache debug logs
        cache_logs = []
        log_summaries = []
        if os.path.exists("/state/cache_logs"):
            cache_logs = [f for f in os.listdir("/state/cache_logs") if f.startswith("unsloth_cache")]
            
            # Read and summarize each log
            for log_file in cache_logs[:3]:  # First 3 logs
                log_path = os.path.join("/state/cache_logs", log_file)
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        # Look for key events
                        summary = {
                            "file": log_file,
                            "total_lines": len(lines),
                            "cache_initialized": any("Cache INITIALIZED" in line for line in lines),
                            "failures": sum(1 for line in lines if "FAILURE" in line),
                            "position_overflows": sum(1 for line in lines if "position overflow" in line),
                        }
                        log_summaries.append(summary)
                except:
                    pass
        
        print(f"\n📋 Cache logs generated: {len(cache_logs)} files")
        for summary in log_summaries:
            print(f"  - {summary['file']}: {summary['total_lines']} lines, "
                  f"initialized={summary['cache_initialized']}, "
                  f"failures={summary['failures']}, "
                  f"overflows={summary['position_overflows']}")
        
        result = {
            "success": True,
            "train_loss": train_result.metrics.get("train_loss") if hasattr(train_result, "metrics") else None,
            "global_step": train_result.global_step if hasattr(train_result, "global_step") else None,
            "cache_logs": cache_logs,
            "log_summaries": log_summaries,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
        }
        
        with open("/state/step6_result.json", "w") as f:
            json.dump(result, f)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}")
        
        # Check if it's the cache position error
        cache_error = "cache_position" in error_msg or "illegal memory access" in error_msg
        
        print("\nMemory at error:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # Check cache logs for debugging info
        cache_logs = []
        if os.path.exists("/state/cache_logs"):
            cache_logs = [f for f in os.listdir("/state/cache_logs") if f.startswith("unsloth_cache")]
            print(f"\n📋 Cache logs available for debugging: {cache_logs}")
        
        result = {
            "success": False,
            "error": error_msg,
            "cache_error": cache_error,
            "cache_logs": cache_logs,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
        }
        
        with open("/state/step6_result.json", "w") as f:
            json.dump(result, f)
        
        return result

@app.function(
    image=image,
    volumes={"/state": state_volume}
)
def get_cache_logs():
    """Download and display cache debug logs"""
    import os
    
    print("="*80)
    print("CACHE DEBUG LOGS")
    print("="*80)
    
    logs_dir = "/state/cache_logs"
    if not os.path.exists(logs_dir):
        return {"error": "No cache logs directory found"}
    
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    logs_content = {}
    for log_file in log_files:
        log_path = os.path.join(logs_dir, log_file)
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                # Truncate if too long
                if len(content) > 50000:
                    content = content[:25000] + "\n\n... [TRUNCATED] ...\n\n" + content[-25000:]
                logs_content[log_file] = content
        except Exception as e:
            logs_content[log_file] = f"Error reading log: {e}"
    
    return logs_content

@app.function(
    image=image,
    volumes={"/state": state_volume}
)
def get_all_results():
    """Get all saved results"""
    import json
    import os
    
    results = {}
    
    # Read all result files
    for step in range(1, 7):
        filename = f"/state/step{step}_result.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results[f"step{step}"] = json.load(f)
    
    return results

# Convenience functions for running steps
@app.local_entrypoint()
def main():
    print("GPT-OSS-20B GRPO Debugging")
    print("This script follows the pattern: SFT -> Save -> Load -> GRPO")
    print("\nRun individual steps with:")
    print("  modal run modal_grpo_gpt_oss.py::step1_check_environment")
    print("  modal run modal_grpo_gpt_oss.py::step2_load_base_model")
    print("  modal run modal_grpo_gpt_oss.py::step3_sft_training")
    print("  modal run modal_grpo_gpt_oss.py::step4_load_sft_model")
    print("  modal run modal_grpo_gpt_oss.py::step5_create_grpo_trainer")
    print("  modal run modal_grpo_gpt_oss.py::step6_run_grpo_training")
    print("\nUtility functions:")
    print("  modal run modal_grpo_gpt_oss.py::get_cache_logs")
    print("  modal run modal_grpo_gpt_oss.py::get_all_results")

if __name__ == "__main__":
    main()