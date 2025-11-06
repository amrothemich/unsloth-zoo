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
                
                # Check if cache_position exceeds sliding window
                if hasattr(cache_position, 'item'):
                    pos_value = cache_position.item()
                elif hasattr(cache_position, '__len__') and len(cache_position) > 0:
                    pos_value = cache_position[0].item() if hasattr(cache_position[0], 'item') else cache_position[0]
                else:
                    pos_value = cache_position
                
                if pos_value >= sliding_window_size:
                    print(f"🔧 PREVENTING cache_position overflow: {pos_value} -> {sliding_window_size - 1}")
                    # Create a corrected cache_position
                    corrected_position = torch.tensor([sliding_window_size - 1], 
                                                     device=cache_position.device, 
                                                     dtype=cache_position.dtype)
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
        
        def safe_init(self, max_cache_len, max_batch_size, heads, head_dim, dtype=torch.float32, device=None):
            # Call original init
            original_init(self, max_cache_len, max_batch_size, heads, head_dim, dtype, device)
            
            # Store the actual window size for bounds checking
            self._actual_window_size = max_cache_len
            print(f"🔧 SlidingWindowLayer initialized with window size: {max_cache_len}")
        
        # Store original update
        original_update = SlidingWindowLayer.update
        
        def bounded_update(self, key_states, value_states, cache_kwargs):
            # Check cache_position bounds before any operations
            cache_position = cache_kwargs.get('cache_position', None)
            if cache_position is not None and hasattr(self, '_actual_window_size'):
                if hasattr(cache_position, 'item'):
                    pos_value = cache_position.item()
                elif hasattr(cache_position, '__len__') and len(cache_position) > 0:
                    pos_value = cache_position[0].item() if hasattr(cache_position[0], 'item') else cache_position[0]
                else:
                    pos_value = cache_position
                
                # Use modulo to wrap around the sliding window
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

TEMPORARY_PATCHES.append(patch_cache_position_generation)
TEMPORARY_PATCHES.append(patch_sliding_window_cache_creation)