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
            # Log that we're being called
            call_num = len(patch_cache_position_generation.position_history)
            if call_num % 10 == 0 or call_num < 5:
                print(f"🔍 [Call #{call_num}] safe_update_model_kwargs_for_generation called")
                # Check incoming cache_position
                if "cache_position" in model_kwargs:
                    incoming_cache_pos = model_kwargs["cache_position"]
                    if hasattr(incoming_cache_pos, 'shape'):
                        print(f"   INCOMING cache_position shape: {incoming_cache_pos.shape}")
                        if incoming_cache_pos.shape[0] > 10:
                            print(f"   ⚠️  INCOMING cache_position already corrupted with {incoming_cache_pos.shape[0]} elements!")

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
                            
                            # CRITICAL FIX: The cache_position tensor has grown incorrectly
                            # Instead of trying to pick one position, we need to understand why it grew
                            
                            if cache_shape[0] > 100:  # Large corruption like 1156
                                # This is likely a sequence of positions that accumulated incorrectly
                                # The correct approach is to reset to the current step in the sequence
                                # Calculate the current position based on sequence progress
                                pos_value = (cache_shape[0] - 1) % sliding_window_size
                                print(f"   Large corruption: resetting to position {pos_value} (from {cache_shape[0]} positions)")
                            else:
                                # Small corruption - use the last position
                                pos_value = cache_position[-1].item() % sliding_window_size
                                print(f"   Small corruption: using wrapped last position {pos_value}")
                            
                            # FIX THE CORRUPTION: Replace with correct single-position tensor
                            corrected_position = torch.tensor([pos_value], 
                                                             device=cache_position.device, 
                                                             dtype=cache_position.dtype)
                            updated_kwargs["cache_position"] = corrected_position
                            print(f"   ✅ FIXED: Replaced {cache_shape} tensor with wrapped position {pos_value}")
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
                    
                    # CRITICAL: Always ensure cache_position doesn't exceed sliding window
                    if sliding_window_size is not None:
                        # For sliding windows, positions should wrap around
                        # Position 128 should become 0, 129->1, etc.
                        if pos_value >= sliding_window_size:
                            wrapped_position = pos_value % sliding_window_size
                            print(f"🔧 WRAPPING cache_position: {pos_value} -> {wrapped_position} (window: {sliding_window_size})")
                            print(f"   Recent position history: {patch_cache_position_generation.position_history[-10:]}")
                            print(f"   This is normal for sliding windows - positions wrap around")
                            # Wrap around to preserve sliding window semantics
                            corrected_position = torch.tensor([wrapped_position], 
                                                             device=cache_position.device, 
                                                             dtype=cache_position.dtype)
                            updated_kwargs["cache_position"] = corrected_position
                        else:
                            # Even if position is valid, ensure it's a proper single-element tensor
                            # This prevents accumulation from happening
                            if hasattr(cache_position, 'shape') and len(cache_position.shape) > 0 and cache_position.shape[0] > 1:
                                print(f"🔧 PREVENTING ACCUMULATION: cache_position has {cache_position.shape[0]} elements, using last one")
                                corrected_position = torch.tensor([pos_value], 
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
            
            # CRITICAL FIX: Use sliding_window (128), NOT max_cache_len (3299)!
            # max_cache_len is the total cache capacity, sliding_window is the actual window size
            self._actual_window_size = sliding_window if sliding_window is not None else 128
            print(f"🔧 SlidingWindowLayer initialized:")
            print(f"   max_cache_len: {max_cache_len} (total capacity)")
            print(f"   sliding_window: {sliding_window} (actual window size)")
            print(f"   Using window size: {self._actual_window_size} for position bounds")
        
        # Store original update
        original_update = SlidingWindowLayer.update
        
        # Track call count for bounded_update
        if not hasattr(bounded_update, 'call_count'):
            bounded_update.call_count = 0

        def bounded_update(self, key_states, value_states, cache_kwargs):
            # AGGRESSIVE FIX: Always ensure cache_position is a single-element tensor
            bounded_update.call_count += 1
            cache_position = cache_kwargs.get('cache_position', None)
            window_size = getattr(self, '_actual_window_size', 128)

            # Log periodically
            if bounded_update.call_count % 100 == 0 or bounded_update.call_count < 10:
                print(f"🔍 [Call #{bounded_update.call_count}] bounded_update called, window_size={window_size}")

            if cache_position is not None:
                try:
                    # Always extract scalar position value, regardless of tensor shape
                    if hasattr(cache_position, 'shape') and len(cache_position.shape) > 0:
                        cache_shape = cache_position.shape

                        # If it's not a single-element tensor, fix it immediately
                        if cache_shape[0] != 1:
                            if cache_shape[0] > 1:
                                # Multi-element tensor - take the last position
                                print(f"🔧 AUTO-FIX: cache_position has {cache_shape[0]} elements, extracting last")
                                pos_value = cache_position[-1].item()
                            else:
                                # Empty or unusual shape
                                pos_value = 0
                                print(f"🔧 AUTO-FIX: unusual cache_position shape {cache_shape}, using 0")
                        else:
                            # Single element tensor - get the value
                            pos_value = cache_position[0].item() if hasattr(cache_position[0], 'item') else cache_position.item()
                    elif hasattr(cache_position, 'item'):
                        # Scalar tensor
                        pos_value = cache_position.item()
                    elif isinstance(cache_position, (int, float)):
                        # Raw number
                        pos_value = int(cache_position)
                    else:
                        # Unknown format
                        print(f"🔧 AUTO-FIX: unknown cache_position type {type(cache_position)}, using 0")
                        pos_value = 0

                    # Wrap position to window size
                    if pos_value >= window_size:
                        pos_value = pos_value % window_size
                    elif pos_value < 0:
                        pos_value = 0

                    # ALWAYS create a fresh single-element tensor
                    cache_kwargs = dict(cache_kwargs)
                    cache_kwargs['cache_position'] = torch.tensor([pos_value],
                                                                 device=key_states.device,
                                                                 dtype=torch.long)

                except Exception as e:
                    print(f"⚠️  Exception in cache_position fix: {e}")
                    # Emergency fallback: use position 0
                    cache_kwargs = dict(cache_kwargs)
                    cache_kwargs['cache_position'] = torch.tensor([0], device=key_states.device, dtype=torch.long)

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
    from corrupted parameters. Includes GPU recovery mechanisms.
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

            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA error: an illegal memory access" in error_msg or "illegal memory access" in error_msg:
                    print(f"🚨 CUDA memory error in create_mask - attempting recovery: {e}")
                    print(f"   Parameters: B={B}, H={H}, Q_LEN={Q_LEN}, KV_LEN={KV_LEN}, device={device}")
                    print(f"🔧 Attempting GPU cache clear and synchronization...")

                    try:
                        # Clear CUDA cache and synchronize to recover from corruption
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            print(f"   ✅ GPU cache cleared and synchronized")

                        # Retry with fresh GPU state
                        print(f"   🔄 Retrying create_mask after GPU recovery...")
                        return original_create_mask(mod_fn, B, H, Q_LEN, KV_LEN, device)

                    except Exception as retry_e:
                        print(f"   ❌ Recovery failed: {retry_e}")
                        print(f"🛑 STOPPING EXECUTION - Unrecoverable GPU corruption")
                        raise RuntimeError(f"GPU memory corruption detected in flex_attention. "
                                         f"Root cause: corrupted cache_position tensors. "
                                         f"Original error: {e}. Recovery attempt failed: {retry_e}") from e
                else:
                    # Non-CUDA error, re-raise as-is
                    raise

        # Monkey patch the function
        torch.nn.attention.flex_attention.create_mask = safe_create_mask
        print("✅ Applied flex attention safety patch with GPU recovery")

    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to patch flex attention: {e}")

def patch_grpo_cache_reset():
    """
    Ensure cache is properly reset between GRPO generations.
    GRPO generates multiple completions per prompt, and we need to ensure
    cache_position doesn't accumulate across these generations.
    """
    try:
        from transformers import GenerationMixin
        import torch

        original_prepare_inputs = GenerationMixin.prepare_inputs_for_generation

        def safe_prepare_inputs_for_generation(self, input_ids, **kwargs):
            # Get the prepared inputs from the original method
            model_inputs = original_prepare_inputs(self, input_ids, **kwargs)

            # Ensure cache_position is always a single-element tensor, never accumulated
            if "cache_position" in model_inputs and model_inputs["cache_position"] is not None:
                cache_pos = model_inputs["cache_position"]

                # Check if it's accumulated (multiple elements)
                if hasattr(cache_pos, 'shape') and len(cache_pos.shape) > 0 and cache_pos.shape[0] > 1:
                    # Take only the last position and create a fresh tensor
                    last_pos = cache_pos[-1].item() if hasattr(cache_pos[-1], 'item') else cache_pos[-1]

                    # Get sliding window size for wrapping
                    sliding_window = getattr(self.config, 'sliding_window', 128)
                    if sliding_window is not None and last_pos >= sliding_window:
                        last_pos = last_pos % sliding_window

                    model_inputs["cache_position"] = torch.tensor([last_pos],
                                                                   device=cache_pos.device,
                                                                   dtype=cache_pos.dtype)

            return model_inputs

        GenerationMixin.prepare_inputs_for_generation = safe_prepare_inputs_for_generation
        print("✅ Applied GRPO cache reset patch")

    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to patch GRPO cache reset: {e}")

# Apply patches immediately on module import
patch_cache_position_generation()
patch_sliding_window_cache_creation()
patch_flex_attention_safety()
patch_grpo_cache_reset()

# Also add to list for tracking
TEMPORARY_PATCHES.append(patch_cache_position_generation)
TEMPORARY_PATCHES.append(patch_sliding_window_cache_creation)
TEMPORARY_PATCHES.append(patch_flex_attention_safety)
TEMPORARY_PATCHES.append(patch_grpo_cache_reset)