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
import os
import datetime
import traceback
import torch

def patch_lightweight_cache_debugging():
    """
    Lightweight cache debugging - minimal overhead during normal operation,
    detailed logging only when issues occur
    """
    try:
        from transformers.cache_utils import SlidingWindowLayer
        
        # Global tracking
        if not hasattr(patch_lightweight_cache_debugging, 'call_count'):
            patch_lightweight_cache_debugging.call_count = 0
            patch_lightweight_cache_debugging.failure_count = 0
            
            # Set up logging
            log_dir = os.environ.get('UNSLOTH_CACHE_DEBUG_LOG_DIR', '.')
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"unsloth_cache_lightweight_{timestamp}.log"
            patch_lightweight_cache_debugging.log_path = os.path.join(log_dir, log_filename)
            
            os.makedirs(log_dir, exist_ok=True)
            
            with open(patch_lightweight_cache_debugging.log_path, 'w') as f:
                f.write(f"Unsloth Lightweight Cache Debug - Started at {datetime.datetime.now()}\n")
                f.write("="*80 + "\n\n")
            
            print(f"📝 Lightweight cache debug logging to: {patch_lightweight_cache_debugging.log_path}")
        
        original_update = SlidingWindowLayer.update
        
        def lightweight_debug_update(self, key_states, value_states, cache_kwargs):
            call_count = patch_lightweight_cache_debugging.call_count
            patch_lightweight_cache_debugging.call_count += 1
            
            # Track cache lifecycle - detect when it should initialize vs when it's stuck
            cache_was_none = not hasattr(self, 'keys') or self.keys is None
            cache_id = id(self)
            
            # Minimal tracking - just call the original method and catch exceptions
            try:
                result = original_update(self, key_states, value_states, cache_kwargs)
                
                # Check if cache state changed after the call
                cache_is_none_after = not hasattr(self, 'keys') or self.keys is None
                
                # Log cache lifecycle changes
                if cache_was_none and not cache_is_none_after:
                    # Cache just got initialized!
                    print(f"🎉 Cache #{call_count}: Cache INITIALIZED (id:{cache_id}) - shape: {self.keys.shape}")
                    with open(patch_lightweight_cache_debugging.log_path, 'a') as f:
                        f.write(f"Call #{call_count}: Cache INITIALIZED (id:{cache_id}) - shape: {self.keys.shape}\n")
                
                # Special logging around the critical call #2595
                # Also log compiler behavior messages
                if call_count == 1:
                    with open(patch_lightweight_cache_debugging.log_path, 'a') as f:
                        f.write("🔍 WATCHING FOR TORCH.COMPILE MESSAGES AROUND FAILURE\n\n")
                
                if 2590 <= call_count <= 2600:
                    with open(patch_lightweight_cache_debugging.log_path, 'a') as f:
                        f.write(f"\n🔍 CRITICAL RANGE - Call #{call_count}:\n")
                        f.write(f"  Cache ID: {cache_id}\n")
                        f.write(f"  Key states shape: {key_states.shape}\n")
                        f.write(f"  Value states shape: {value_states.shape}\n")
                        f.write(f"  Cache kwargs: {cache_kwargs}\n")
                        if hasattr(self, 'keys') and self.keys is not None:
                            f.write(f"  Cache keys shape: {self.keys.shape}\n")
                            f.write(f"  Cache keys device: {self.keys.device}\n")
                            f.write(f"  Cache keys dtype: {self.keys.dtype}\n")
                            f.write(f"  Cache keys contiguous: {self.keys.is_contiguous()}\n")
                        f.write(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB\n")
                        f.write("\n")
                
                elif not cache_was_none and cache_is_none_after:
                    # Cache got reset/cleared - this is suspicious!
                    print(f"⚠️  Cache #{call_count}: Cache RESET TO NONE (id:{cache_id}) - this might be a problem!")
                    with open(patch_lightweight_cache_debugging.log_path, 'a') as f:
                        f.write(f"Call #{call_count}: Cache RESET TO NONE (id:{cache_id}) - SUSPICIOUS!\n")
                
                # Periodic progress (minimal overhead)
                if call_count % 10000 == 0:
                    cache_status = "None" if cache_is_none_after else f"Initialized(shape:{self.keys.shape})"
                    print(f"✅ Cache #{call_count}: OK - Cache: {cache_status}")
                    with open(patch_lightweight_cache_debugging.log_path, 'a') as f:
                        f.write(f"Progress: Call #{call_count} - Cache: {cache_status}\n")
                
                return result
                
            except Exception as e:
                # Only do detailed debugging when there's an actual failure
                patch_lightweight_cache_debugging.failure_count += 1
                
                print(f"❌ Cache #{call_count}: FAILED - {type(e).__name__}: {e}")
                
                # Now do the comprehensive debugging for this failure
                with open(patch_lightweight_cache_debugging.log_path, 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"FAILURE #{patch_lightweight_cache_debugging.failure_count} at call #{call_count}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Error: {type(e).__name__}: {e}\n\n")
                    
                    # Device info
                    f.write(f"Device info:\n")
                    f.write(f"  self.device: {getattr(self, 'device', 'MISSING')}\n")
                    f.write(f"  key_states.device: {key_states.device}\n")
                    f.write(f"  key_states.shape: {key_states.shape}\n")
                    
                    # Cache state
                    f.write(f"Cache state:\n")
                    if hasattr(self, 'keys'):
                        if self.keys is not None:
                            f.write(f"  keys.shape: {self.keys.shape}\n")
                        else:
                            f.write(f"  keys: None\n")
                    else:
                        f.write(f"  keys: MISSING\n")
                    
                    # Memory state
                    if torch.cuda.is_available():
                        f.write(f"GPU Memory:\n")
                        f.write(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
                    
                    # Stack trace
                    f.write(f"\nFull traceback:\n")
                    traceback.print_exc(file=f)
                    f.write("\n" + "="*80 + "\n\n")
                    f.flush()
                
                # Check for CUDA illegal memory access
                if "illegal memory access" in str(e):
                    print(f"🚨 CUDA ILLEGAL MEMORY ACCESS detected at call #{call_count}!")
                    print(f"📝 Full details in: {patch_lightweight_cache_debugging.log_path}")
                
                # Return fallback
                return key_states, value_states
        
        # Apply the patch
        SlidingWindowLayer.update = lightweight_debug_update
        print(f"✅ Applied lightweight cache debugging (minimal overhead)")
        
    except Exception as e:
        print(f"❌ Failed to apply lightweight cache debugging: {e}")

# Add to temporary patches
TEMPORARY_PATCHES.append(patch_lightweight_cache_debugging)