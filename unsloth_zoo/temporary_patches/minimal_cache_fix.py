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

def patch_minimal_sliding_window_cache_device_fix():
    """
    Minimal cache fix that ONLY fixes device issues without interfering 
    with normal cache initialization and operation.
    """
    try:
        from transformers.cache_utils import SlidingWindowLayer
        import torch
        
        # Store the original update method
        original_update = SlidingWindowLayer.update
        
        def device_safe_update(self, key_states, value_states, cache_kwargs):
            # ONLY fix: Ensure device is set before any device-dependent operations
            if not hasattr(self, 'device') or self.device is None:
                self.device = key_states.device
            
            # Try the original method first
            try:
                return original_update(self, key_states, value_states, cache_kwargs)
            except RuntimeError as e:
                if "device" in str(e) and "tensor" in str(e):
                    # This is likely the device issue - try to fix by ensuring device consistency
                    print(f"🔧 Fixing device issue in cache update: {e}")
                    
                    # If we have cache tensors, make sure they're on the right device
                    if hasattr(self, 'keys') and self.keys is not None:
                        if self.keys.device != key_states.device:
                            print(f"🔧 Moving cache from {self.keys.device} to {key_states.device}")
                            self.keys = self.keys.to(key_states.device)
                            self.values = self.values.to(value_states.device)
                    
                    # Ensure self.device matches
                    self.device = key_states.device
                    
                    # Try again
                    return original_update(self, key_states, value_states, cache_kwargs)
                else:
                    # Different error - re-raise
                    raise
        
        # Apply the minimal patch
        SlidingWindowLayer.update = device_safe_update
        print("✅ Applied minimal cache device fix (preserves normal initialization)")
        
    except ImportError:
        # transformers version might not have SlidingWindowLayer
        pass
    except Exception as e:
        print(f"Warning: Failed to patch SlidingWindowLayer minimally: {e}")

TEMPORARY_PATCHES.append(patch_minimal_sliding_window_cache_device_fix)