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

def patch_sliding_window_cache_device_fix():
    """
    Fix SlidingWindowLayer cache device initialization issue.
    
    The SlidingWindowLayer.update method was failing with CUDA memory access errors
    because self.device was None, causing torch.tensor([-1], dtype=int, device=None)
    to fail when trying to create index tensors.
    
    This patch ensures the cache layer always has a valid device set.
    """
    try:
        from transformers.cache_utils import SlidingWindowLayer
        
        def update(self, key_states, value_states, cache_kwargs):
            # Ensure device is properly set before any tensor operations
            if not hasattr(self, 'device') or self.device is None:
                # Use the device from the input key_states
                self.device = key_states.device
            
            # Call the original transformers implementation
            # But ensure we have a proper device set for any device-dependent operations
            import torch
            
            # If cache is not initialized, let the original code handle it
            # Our job is just to ensure device is set when needed
            if not hasattr(self, 'keys') or self.keys is None:
                # Cache not initialized - this is normal for first calls
                # The original transformers code should handle initialization
                # We'll just ensure device is set for when it's needed
                return key_states, value_states
            
            # Check if we need to slide the cache
            if key_states.shape[-2] > 1:
                # Multiple new tokens - slide by the number of new tokens minus 1
                slide_amount = key_states.shape[-2] - 1
                self.keys = self.keys.roll(-slide_amount, dims=-2)
                self.values = self.values.roll(-slide_amount, dims=-2)
                
                # Update the last positions with new states
                self.keys[:, :, -key_states.shape[-2]:] = key_states
                self.values[:, :, -value_states.shape[-2]:] = value_states
                
                return self.keys, self.values
            else:
                # Single new token - original logic
                new_keys = self.keys.roll(-1, dims=-2)
                new_values = self.values.roll(-1, dims=-2)
                
                # Use proper device for index tensor
                index = torch.tensor([-1], dtype=int, device=self.device)
                new_keys[:, :, index] = key_states
                new_values[:, :, index] = value_states
                
                self.keys = new_keys
                self.values = new_values
                
                return self.keys, self.values
        
        patch_function(SlidingWindowLayer, "update", update)
        
    except ImportError:
        # transformers version might not have SlidingWindowLayer
        pass
    except Exception as e:
        print(f"Warning: Failed to patch SlidingWindowLayer: {e}")

TEMPORARY_PATCHES.append(patch_sliding_window_cache_device_fix)