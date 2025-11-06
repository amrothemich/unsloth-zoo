# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .common import TEMPORARY_PATCHES
from .utils import patch_function
import torch

def patch_dtype_consistency():
    """
    Fix dtype consistency issues in mixed precision training.
    Ensures all matrix operations use consistent dtypes (BFloat16).
    """
    try:
        # Import the modules that commonly have dtype issues
        import torch.nn as nn
        import transformers
        from unsloth import FastLanguageModel
        
        print("🔧 Applying dtype consistency patches...")
        
        # Store original forward methods
        original_linear_forward = nn.Linear.forward
        
        def safe_linear_forward(self, input):
            """Ensure both weight and input have same dtype for Linear layers"""
            # If weight is bfloat16, ensure input is also bfloat16
            if self.weight.dtype == torch.bfloat16 and input.dtype != torch.bfloat16:
                input = input.to(torch.bfloat16)
            # If weight is float32, ensure input is also float32
            elif self.weight.dtype == torch.float32 and input.dtype != torch.float32:
                input = input.to(torch.float32)
            
            return original_linear_forward(self, input)
        
        # Apply the patch
        nn.Linear.forward = safe_linear_forward
        
        print("✅ Applied dtype consistency patch for Linear layers")
        
        # Also patch attention modules which commonly have dtype issues
        try:
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
            
            original_attention_forward = GptOssAttention.forward
            
            def safe_attention_forward(self, *args, **kwargs):
                """Ensure attention computations use consistent dtypes"""
                # Ensure all inputs to attention are in the same dtype as the model
                if hasattr(self, 'q_proj') and hasattr(self.q_proj, 'weight'):
                    target_dtype = self.q_proj.weight.dtype
                    
                    # Convert input tensors to target dtype
                    new_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor) and arg.dtype != target_dtype:
                            new_args.append(arg.to(target_dtype))
                        else:
                            new_args.append(arg)
                    
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.dtype != target_dtype:
                            new_kwargs[k] = v.to(target_dtype)
                        else:
                            new_kwargs[k] = v
                    
                    return original_attention_forward(self, *new_args, **new_kwargs)
                
                return original_attention_forward(self, *args, **kwargs)
            
            GptOssAttention.forward = safe_attention_forward
            print("✅ Applied dtype consistency patch for GptOssAttention")
            
        except ImportError:
            print("ℹ️  GptOssAttention not available, skipping attention dtype patch")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to apply dtype consistency patches: {e}")
        return False

# Register the patch
TEMPORARY_PATCHES.append(patch_dtype_consistency)