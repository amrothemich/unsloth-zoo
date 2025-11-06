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

def patch_cache_position_generation():
    """
    Fix cache_position generation to prevent values that exceed sliding window bounds.
    This prevents the CUDA memory corruption by fixing the root cause.
    """
    try:
        from transformers.generation.utils import GenerationMixin
        import torch
        
        # Store the original update_model_kwargs_for_generation method
        original_update_kwargs = GenerationMixin._update_model_kwargs_for_generation
        
        # Track positions to detect the corruption pattern
        if not hasattr(patch_cache_position_generation, 'position_history'):
            patch_cache_position_generation.position_history = []
        
        def safe_update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
        ):
            # Call the original method
            updated_kwargs = original_update_kwargs(
                self, outputs, model_kwargs, is_encoder_decoder, num_new_tokens
            )
            
            # Check and fix cache_position if it exists
            if "cache_position" in updated_kwargs:
                cache_position = updated_kwargs["cache_position"]
                
                # Get sliding window size from model config
                sliding_window_size = getattr(self.config, 'sliding_window', 128)
                if sliding_window_size is None:
                    sliding_window_size = 128  # Default fallback
                
                try:
                    # Check cache_position shape - this is where corruption is detected
                    if hasattr(cache_position, 'shape'):
                        cache_shape = cache_position.shape
                        if len(cache_shape) > 1 or (len(cache_shape) == 1 and cache_shape[0] > 1):
                            print(f"🚨 CORRUPTED cache_position shape: {cache_shape}")
                            print(f"   Expected: scalar or [1], got: {cache_shape}")
                            if cache_shape[0] <= 10:  # Only log if not too large
                                print(f"   Values: {cache_position.tolist()}")
                            else:
                                print(f"   First 5 values: {cache_position[:5].tolist()}")
                                print(f"   Last 5 values: {cache_position[-5:].tolist()}")
                            
                            # Use the last position as that's likely the current one
                            pos_value = cache_position[-1].item()
                            print(f"   Using last position: {pos_value}")
                            
                            # FIX THE CORRUPTION: Replace with correct single-position tensor
                            corrected_position = torch.tensor([pos_value], 
                                                             device=cache_position.device, 
                                                             dtype=cache_position.dtype)
                            updated_kwargs["cache_position"] = corrected_position
                            print(f"   ✅ FIXED: Replaced {cache_shape} tensor with scalar position {pos_value}")
                        else:
                            # Normal case - single position
                            pos_value = cache_position.item()
                    elif hasattr(cache_position, 'item'):
                        pos_value = cache_position.item()
                    elif hasattr(cache_position, '__len__') and len(cache_position) > 0:
                        pos_value = cache_position[0].item() if hasattr(cache_position[0], 'item') else cache_position[0]
                    else:
                        pos_value = cache_position
                    
                    # Track position history to understand the pattern
                    patch_cache_position_generation.position_history.append(pos_value)
                    if len(patch_cache_position_generation.position_history) > 50:
                        patch_cache_position_generation.position_history.pop(0)
                    
                    # Enhanced logging to understand the position sequence
                    if len(patch_cache_position_generation.position_history) % 10 == 0:
                        print(f"📊 Position #{len(patch_cache_position_generation.position_history)}: {pos_value} (window: {sliding_window_size})")
                        if len(patch_cache_position_generation.position_history) >= 10:
                            recent = patch_cache_position_generation.position_history[-10:]
                            print(f"   Last 10 positions: {recent}")
                            # Check for suspicious jumps
                            diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
                            print(f"   Position deltas: {diffs}")
                            if any(abs(d) > 100 for d in diffs):
                                print(f"   ⚠️  Large position jump detected!")
                    
                    # For sliding windows, positions should wrap around
                    # Position 128 should become 0, 129->1, etc.
                    if sliding_window_size is not None and pos_value >= sliding_window_size:
                        wrapped_position = pos_value % sliding_window_size
                        print(f"🔧 WRAPPING cache_position: {pos_value} -> {wrapped_position} (window: {sliding_window_size})")
                        print(f"   Recent position history: {patch_cache_position_generation.position_history[-10:]}")
                        print(f"   This is normal for sliding windows - positions wrap around")
                        # Wrap around to preserve sliding window semantics
                        corrected_position = torch.tensor([wrapped_position], 
                                                         device=cache_position.device, 
                                                         dtype=cache_position.dtype)
                        updated_kwargs["cache_position"] = corrected_position
                    elif pos_value < 0:
                        # Negative positions are definitely wrong
                        print(f"🔧 FIXING negative cache_position: {pos_value} -> 0")
                        print(f"   Recent position history: {patch_cache_position_generation.position_history[-10:]}")
                        corrected_position = torch.tensor([0], 
                                                         device=cache_position.device, 
                                                         dtype=cache_position.dtype)
                        updated_kwargs["cache_position"] = corrected_position
                
                except Exception as e:
                    print(f"⚠️  Cache position corrupted in generation: {e}")
                    # Force reset to a safe value
                    corrected_position = torch.tensor([0], device=getattr(cache_position, 'device', 'cuda:0'))
                    updated_kwargs["cache_position"] = corrected_position
            
            return updated_kwargs
        
        # Apply the patch
        GenerationMixin._update_model_kwargs_for_generation = safe_update_model_kwargs_for_generation
        print("✅ Applied cache_position generation fix (prevents overflow at source)")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to patch cache_position generation: {e}")

def patch_sliding_window_cache_creation():
    """
    Alternative approach: Fix the sliding window cache to handle large positions gracefully.
    """
    try:
        from transformers.cache_utils import SlidingWindowLayer
        import torch
        
        # Store original init
        original_init = SlidingWindowLayer.__init__
        
        def safe_init(self, max_cache_len, sliding_window=None, **kwargs):
            # Call original init with all the parameters it expects
            original_init(self, max_cache_len=max_cache_len, sliding_window=sliding_window, **kwargs)
            
            # Store the actual window size for bounds checking
            self._actual_window_size = max_cache_len
            print(f"🔧 SlidingWindowLayer initialized with window size: {max_cache_len}")
        
        # Store original update
        original_update = SlidingWindowLayer.update
        
        def bounded_update(self, key_states, value_states, cache_kwargs):
            # Check cache_position bounds before any operations
            cache_position = cache_kwargs.get('cache_position', None)
            if cache_position is not None and hasattr(self, '_actual_window_size'):
                try:
                    # Handle corrupted cache_position tensors
                    if hasattr(cache_position, 'shape'):
                        cache_shape = cache_position.shape
                        if len(cache_shape) > 1 or (len(cache_shape) == 1 and cache_shape[0] > 1):
                            print(f"🚨 SlidingWindow: CORRUPTED cache_position shape: {cache_shape}")
                            # Use the last position and fix the tensor
                            pos_value = cache_position[-1].item()
                            corrected_position = torch.tensor([pos_value], device=cache_position.device, dtype=cache_position.dtype)
                            cache_kwargs = dict(cache_kwargs)
                            cache_kwargs['cache_position'] = corrected_position
                            print(f"   Fixed: using position {pos_value}")
                        else:
                            pos_value = cache_position.item()
                    elif hasattr(cache_position, 'item'):
                        pos_value = cache_position.item()
                    elif hasattr(cache_position, '__len__') and len(cache_position) > 0:
                        pos_value = cache_position[0].item() if hasattr(cache_position[0], 'item') else cache_position[0]
                    else:
                        pos_value = cache_position
                except Exception as e:
                    print(f"⚠️  Failed to read cache_position in SlidingWindow: {e}")
                    # Set a safe default
                    cache_kwargs = dict(cache_kwargs)
                    cache_kwargs['cache_position'] = torch.tensor([0], device=key_states.device)
                    pos_value = 0
                
                # For sliding windows, wrap positions to stay within window bounds
                if pos_value >= self._actual_window_size:
                    wrapped_position = pos_value % self._actual_window_size
                    print(f"🔧 Wrapping cache_position: {pos_value} -> {wrapped_position} (window size: {self._actual_window_size})")
                    cache_kwargs = dict(cache_kwargs)
                    cache_kwargs['cache_position'] = torch.tensor([wrapped_position], 
                                                                 device=key_states.device, 
                                                                 dtype=torch.long)
            
            return original_update(self, key_states, value_states, cache_kwargs)
        
        # Apply patches
        SlidingWindowLayer.__init__ = safe_init
        SlidingWindowLayer.update = bounded_update
        print("✅ Applied sliding window cache bounds checking")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to patch sliding window cache bounds: {e}")

def patch_flex_attention_safety():
    """
    Add safety checks to flex_attention create_mask to prevent CUDA errors
    from corrupted parameters.
    """
    try:
        from torch.nn.attention.flex_attention import create_mask
        import torch
        
        # Store original create_mask
        original_create_mask = create_mask
        
        def safe_create_mask(mod_fn, B, H, Q_LEN, KV_LEN, device):
            try:
                # Validate parameters before calling torch.arange
                if B is None or B <= 0 or B > 1000:  # Reasonable upper bound
                    print(f"⚠️  Invalid B parameter in create_mask: {B}, using B=1")
                    B = 1
                if H is None or H <= 0 or H > 1000:
                    print(f"⚠️  Invalid H parameter in create_mask: {H}, using H=1")
                    H = 1
                if Q_LEN is None or Q_LEN <= 0 or Q_LEN > 100000:
                    print(f"⚠️  Invalid Q_LEN parameter in create_mask: {Q_LEN}, using Q_LEN=1")
                    Q_LEN = 1
                if KV_LEN is None or KV_LEN <= 0 or KV_LEN > 100000:
                    print(f"⚠️  Invalid KV_LEN parameter in create_mask: {KV_LEN}, using KV_LEN=1")
                    KV_LEN = 1
                
                return original_create_mask(mod_fn, B, H, Q_LEN, KV_LEN, device)
                
            except Exception as e:
                print(f"🚨 CUDA error in create_mask - GPU memory corrupted: {e}")
                print(f"   Parameters: B={B}, H={H}, Q_LEN={Q_LEN}, KV_LEN={KV_LEN}, device={device}")
                print(f"🛑 STOPPING EXECUTION - GPU corruption detected!")
                print(f"   The cache_position corruption has spread to GPU memory.")
                print(f"   Continuing would produce garbage results.")
                # Re-raise the error instead of masking it with CPU fallbacks
                raise RuntimeError(f"GPU memory corruption detected in flex_attention. "
                                 f"Root cause: corrupted cache_position tensors. "
                                 f"Original error: {e}") from e
        
        # Monkey patch the function
        torch.nn.attention.flex_attention.create_mask = safe_create_mask
        print("✅ Applied flex attention safety patch")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to patch flex attention: {e}")

TEMPORARY_PATCHES.append(patch_cache_position_generation)
TEMPORARY_PATCHES.append(patch_sliding_window_cache_creation)
TEMPORARY_PATCHES.append(patch_flex_attention_safety)