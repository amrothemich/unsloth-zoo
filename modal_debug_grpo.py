#!/usr/bin/env python3
"""
Modal script to reproduce and debug GRPO cache position issues.
This gives us direct access to logs and faster iteration.
"""
import modal
from pathlib import Path

# Create Modal app
app = modal.App("debug-grpo-cache")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.46.2",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "xformers",
        "triton",
    )
    .run_commands(
        # Install unsloth from main
        "pip install unsloth@git+https://github.com/unslothai/unsloth.git@main",
        # Install unsloth-zoo from our fix-grpo branch
        "pip install --force-reinstall git+https://github.com/amrothemich/unsloth-zoo.git@fix-grpo",
    )
)

# Volume for persistent storage
volume = modal.Volume.from_name("grpo-debug-volume", create_if_missing=True)

@app.function(
    gpu=modal.gpu.A100(size="40GB"),  # Start with 40GB, can increase if needed
    image=image,
    timeout=1800,  # 30 minutes
    volumes={"/data": volume},
    env={
        "UNSLOTH_CACHE_DEBUG_LOG_DIR": "/data/logs",
        "TORCH_USE_CUDA_DSA": "1",  # Enable device-side assertions
        "CUDA_LAUNCH_BLOCKING": "1",  # Make CUDA operations synchronous for better debugging
    }
)
def debug_grpo_training():
    import os
    import sys
    import torch
    from datetime import datetime
    
    # Create log directory
    os.makedirs("/data/logs", exist_ok=True)
    
    print(f"🚀 Starting GRPO debug run at {datetime.now()}")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🎯 CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
    
    # Import to trigger our patches
    import unsloth_zoo
    print(f"✅ Unsloth Zoo patches loaded")
    
    # Check if grpo_unsloth.py exists locally, if not create a minimal reproducer
    script_path = Path("grpo_unsloth.py")
    if script_path.exists():
        print(f"📄 Found grpo_unsloth.py, executing...")
        # Run the actual script
        exec(open("grpo_unsloth.py").read())
    else:
        print("📄 Creating minimal reproducer...")
        # Minimal reproducer based on the patterns we've seen
        from unsloth import FastLanguageModel
        from unsloth import GRPOTrainer, GRPOConfig
        import torch
        from transformers import TextStreamer
        
        # Load model - using GPT-OSS-20B as specified
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="MidnightSociety/GPT-OSS-20B",
            max_seq_length=2048,  # Can adjust based on needs
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        # Prepare for training
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Create dummy data for testing
        from datasets import Dataset
        
        # Create prompts that will trigger the issue
        # Based on logs, we need prompts that generate ~63 tokens
        dummy_data = {
            "prompt": ["Write a story about a robot: "] * 10,
            "completion": ["Once upon a time, there was a robot who lived in a factory. " * 3] * 10,
        }
        dataset = Dataset.from_dict(dummy_data)
        
        print(f"📊 Dataset size: {len(dataset)}")
        
        # Configure training
        training_args = GRPOConfig(
            output_dir="/data/outputs",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=1000,
            learning_rate=2e-5,
            warmup_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            # GRPO specific
            num_generations=1,
            temperature=0.7,
            max_new_tokens=128,  # This should trigger the sliding window issue
        )
        
        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            config=training_args,
        )
        
        print("🏃 Starting training to reproduce issue...")
        try:
            trainer.train()
            print("✅ Training completed without errors!")
        except Exception as e:
            print(f"❌ Error occurred: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save detailed logs
            log_file = f"/data/logs/crash_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_file, "w") as f:
                f.write(f"Error: {type(e).__name__}: {e}\n\n")
                f.write(traceback.format_exc())
            print(f"💾 Saved crash report to: {log_file}")
    
    # List all log files
    print("\n📋 Log files generated:")
    for f in Path("/data/logs").glob("*"):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    return "Debug run completed"

@app.local_entrypoint()
def main():
    # Copy local grpo_unsloth.py if it exists
    local_script = Path("grpo_unsloth.py")
    if local_script.exists():
        print(f"📤 Uploading {local_script.name} to Modal...")
        with open(local_script, "rb") as f:
            content = f.read()
        
        # Upload to Modal function
        with app.function.remote() as remote:
            remote.write_file("grpo_unsloth.py", content)
    
    # Run the debug function
    result = debug_grpo_training.remote()
    print(f"\n🎯 Result: {result}")
    
    # Download logs
    print("\n📥 Downloading logs...")
    os.makedirs("modal_logs", exist_ok=True)
    
    # Use Modal's volume download functionality
    volume.download("/data/logs", "modal_logs")
    print(f"✅ Logs downloaded to ./modal_logs/")

if __name__ == "__main__":
    main()