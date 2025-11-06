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
import traceback
import gc
import os
import datetime
import io
import sys

def patch_comprehensive_cache_debugging():
    """
    Comprehensive cache debugging to track memory corruption progression
    Logs to file specified by UNSLOTH_CACHE_DEBUG_LOG_DIR environment variable
    """
    try:
        from transformers.cache_utils import SlidingWindowLayer
        import torch
        
        # Global counter and state tracking
        if not hasattr(patch_comprehensive_cache_debugging, 'call_count'):
            patch_comprehensive_cache_debugging.call_count = 0
            patch_comprehensive_cache_debugging.failure_count = 0
            patch_comprehensive_cache_debugging.last_successful_state = {}
            patch_comprehensive_cache_debugging.device_history = []
            
            # Set up logging
            log_dir = os.environ.get('UNSLOTH_CACHE_DEBUG_LOG_DIR', '.')
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"unsloth_cache_debug_{timestamp}.log"
            patch_comprehensive_cache_debugging.log_path = os.path.join(log_dir, log_filename)
            
            # Create log directory if needed
            os.makedirs(log_dir, exist_ok=True)
            
            # Initial log message
            with open(patch_comprehensive_cache_debugging.log_path, 'w') as f:
                f.write(f"Unsloth Cache Debug Log - Started at {datetime.datetime.now()}\n")
                f.write(f"Log path: {patch_comprehensive_cache_debugging.log_path}\n")
                f.write(f"Set UNSLOTH_CACHE_DEBUG_LOG_DIR env var to change log location\n")
                f.write("="*80 + "\n\n")
            
            print(f"📝 Unsloth cache debug logging to: {patch_comprehensive_cache_debugging.log_path}")
        
        original_update = SlidingWindowLayer.update
        
        def log_debug(message):
            """Helper to write to log file"""
            with open(patch_comprehensive_cache_debugging.log_path, 'a') as f:
                f.write(message + '\n')
                f.flush()  # Ensure immediate write
        
        def comprehensive_debug_update(self, key_states, value_states, cache_kwargs):
            call_count = patch_comprehensive_cache_debugging.call_count
            patch_comprehensive_cache_debugging.call_count += 1

            # Log every call to verify patch is working
            if call_count % 100 == 0 or call_count < 5:
                try:
                    with open('/dbfs/FileStore/payer_ai/NLP/CALLM/cache_debug.txt', 'a') as f:
                        f.write(f"✓ SlidingWindowLayer.update call #{call_count}\n")
                except:
                    pass

            # ========================================================================
            # CRITICAL FIX: Check and fix cache_position in cache_kwargs RIGHT NOW!
            # ========================================================================
            if "cache_position" in cache_kwargs and cache_kwargs["cache_position"] is not None:
                cache_pos = cache_kwargs["cache_position"]
                if hasattr(cache_pos, 'shape') and len(cache_pos.shape) > 0 and cache_pos.shape[0] > 1:
                    # CORRUPTED! Fix it immediately
                    debug_msg = f"""
🚨 CACHE CORRUPTION DETECTED IN SlidingWindowLayer.update!
Call #{call_count}
Shape: {cache_pos.shape}
Values (first 20): {cache_pos[:20].tolist() if cache_pos.shape[0] >= 20 else cache_pos.tolist()}
"""
                    try:
                        with open('/dbfs/FileStore/payer_ai/NLP/CALLM/cache_debug.txt', 'a') as f:
                            f.write(debug_msg)
                            f.write('\n' + '='*80 + '\n')
                    except:
                        pass

                    # FIX IT
                    last_pos = cache_pos[-1].item()
                    sliding_window = getattr(self, 'sliding_window', None)
                    if sliding_window is not None and last_pos >= sliding_window:
                        last_pos = last_pos % sliding_window

                    cache_kwargs["cache_position"] = torch.tensor([last_pos],
                                                                   device=cache_pos.device,
                                                                   dtype=cache_pos.dtype)

                    try:
                        with open('/dbfs/FileStore/payer_ai/NLP/CALLM/cache_debug.txt', 'a') as f:
                            f.write(f"✅ FIXED in SlidingWindowLayer.update: Reduced to position {last_pos}\n\n")
                    except:
                        pass

            # Buffer for all debug output
            debug_buffer = io.StringIO()
            
            def debug_print(msg=""):
                """Print to buffer instead of stdout"""
                debug_buffer.write(str(msg) + '\n')
            
            debug_print(f"\n{'='*80}")
            debug_print(f"🔍 CACHE DEBUG #{call_count} - SlidingWindowLayer.update")
            debug_print(f"{'='*80}")
            
            # Get calling context
            stack = traceback.extract_stack()
            calling_context = []
            for frame in reversed(stack[:-1]):  # Skip current frame
                if 'transformers' in frame.filename or 'unsloth' in frame.filename:
                    calling_context.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
                    if len(calling_context) >= 5:  # Show last 5 relevant frames
                        break
            
            debug_print("📍 Call Stack:")
            for ctx in calling_context:
                debug_print(ctx)
            
            # Detailed state inspection
            debug_print(f"\n📊 State Information:")
            debug_print(f"  self.device: {getattr(self, 'device', 'MISSING')}")
            debug_print(f"  self.device type: {type(getattr(self, 'device', None))}")
            debug_print(f"  self.device id: {id(getattr(self, 'device', None))}")
            debug_print(f"  key_states.device: {key_states.device}")
            debug_print(f"  key_states.shape: {key_states.shape}")
            debug_print(f"  key_states.dtype: {key_states.dtype}")
            debug_print(f"  key_states.is_contiguous: {key_states.is_contiguous()}")
            
            # Memory state
            if torch.cuda.is_available():
                debug_print(f"\n💾 GPU Memory State:")
                debug_print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                debug_print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                debug_print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                
                # Check for memory pressure
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = total_memory - torch.cuda.memory_allocated()
                debug_print(f"  Free Memory: {free_memory / 1e9:.2f} GB ({free_memory/total_memory*100:.1f}%)")
            
            # Cache state inspection
            debug_print(f"\n📦 Cache State:")
            if hasattr(self, 'keys'):
                if self.keys is not None:
                    debug_print(f"  Keys shape: {self.keys.shape}")
                    debug_print(f"  Keys device: {self.keys.device}")
                    debug_print(f"  Keys dtype: {self.keys.dtype}")
                    debug_print(f"  Keys contiguous: {self.keys.is_contiguous()}")
                    debug_print(f"  Keys data_ptr: {self.keys.data_ptr()}")
                else:
                    debug_print(f"  ❌ CRITICAL: self.keys is None!")
            else:
                debug_print(f"  ❌ CRITICAL: self.keys attribute missing!")
                
            if hasattr(self, 'values'):
                if self.values is not None:
                    debug_print(f"  Values shape: {self.values.shape}")
                    debug_print(f"  Values device: {self.values.device}")
                else:
                    debug_print(f"  ❌ CRITICAL: self.values is None!")
            else:
                debug_print(f"  ❌ CRITICAL: self.values attribute missing!")
                
            # Check for NaN or Inf if cache exists
            if hasattr(self, 'keys') and self.keys is not None:
                try:
                    has_nan = torch.isnan(self.keys).any().item()
                    has_inf = torch.isinf(self.keys).any().item()
                    debug_print(f"  Keys has NaN: {has_nan}")
                    debug_print(f"  Keys has Inf: {has_inf}")
                except Exception as e:
                    debug_print(f"  ❌ Cannot check NaN/Inf: {e}")
            
            # Device history tracking
            current_device = getattr(self, 'device', None)
            patch_comprehensive_cache_debugging.device_history.append({
                'call': call_count,
                'device': current_device,
                'key_device': key_states.device
            })
            
            # Check for device issues
            device_ok = True
            if not hasattr(self, 'device'):
                debug_print(f"\n🔧 CRITICAL: self.device missing! Setting to key_states.device")
                self.device = key_states.device
                device_ok = False
            elif self.device is None:
                debug_print(f"\n🔧 CRITICAL: self.device is None! Setting to key_states.device")
                self.device = key_states.device
                device_ok = False
            elif isinstance(self.device, int):
                debug_print(f"\n🔧 Converting integer device {self.device} to cuda:{self.device}")
                self.device = f"cuda:{self.device}"
                device_ok = False
            
            # Test the problematic operation
            debug_print(f"\n🧪 Testing index tensor creation...")
            test_success = False
            try:
                test_index = torch.tensor([-1], dtype=int, device=self.device)
                debug_print(f"  ✅ Test index creation successful")
                test_success = True
                
                # Additional memory access test
                if hasattr(self, 'keys') and self.keys is not None:
                    debug_print(f"  🧪 Testing cache memory access...")
                    _ = self.keys[0, 0, 0, 0].item()  # Try to access first element
                    debug_print(f"  ✅ Cache memory access successful")
                    
                    # Test roll operation
                    debug_print(f"  🧪 Testing roll operation...")
                    _ = self.keys.roll(-1, dims=-2)
                    debug_print(f"  ✅ Roll operation successful")
                    
            except Exception as e:
                debug_print(f"  ❌ Test failed: {type(e).__name__}: {e}")
                patch_comprehensive_cache_debugging.failure_count += 1
                
                if "illegal memory access" in str(e):
                    debug_print(f"\n🚨 MEMORY CORRUPTION DETECTED!")
                    debug_print(f"  This is failure #{patch_comprehensive_cache_debugging.failure_count}")
                    debug_print(f"  Total calls before failure: {call_count}")
                    
                    # Compare with last successful state
                    if patch_comprehensive_cache_debugging.last_successful_state:
                        debug_print(f"\n📊 Comparison with last successful state:")
                        last = patch_comprehensive_cache_debugging.last_successful_state
                        debug_print(f"  Last device: {last.get('device')} -> Current: {self.device}")
                        debug_print(f"  Last allocated: {last.get('allocated', 0):.2f} GB -> Current: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                        debug_print(f"  Memory increase: {(torch.cuda.memory_allocated() - last.get('allocated', 0)) / 1e9:.2f} GB")
                    
                    # Show recent device history
                    debug_print(f"\n📜 Recent device history:")
                    for entry in patch_comprehensive_cache_debugging.device_history[-5:]:
                        debug_print(f"  Call #{entry['call']}: device={entry['device']}, key_device={entry['key_device']}")
                    
                    # Emergency fix attempt
                    debug_print(f"\n🚑 Attempting emergency fixes...")
                    
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    debug_print(f"  ✅ Forced garbage collection")
                    
                    # Reset device
                    self.device = key_states.device
                    debug_print(f"  ✅ Reset device to {self.device}")
                    
                    # Try again
                    try:
                        test_index = torch.tensor([-1], dtype=int, device=self.device)
                        debug_print(f"  ✅ Emergency fix successful!")
                        test_success = True
                    except Exception as e2:
                        debug_print(f"  ❌ Emergency fix failed: {e2}")
            
            # Store successful state
            if test_success and device_ok:
                patch_comprehensive_cache_debugging.last_successful_state = {
                    'call': call_count,
                    'device': self.device,
                    'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'cache_shape': self.keys.shape if hasattr(self, 'keys') and self.keys is not None else None
                }
            
            # Call original update
            debug_print(f"\n🔄 Calling original update method...")
            result = None
            update_exception = None
            try:
                result = original_update(self, key_states, value_states, cache_kwargs)
                debug_print(f"✅ Original update successful")
            except Exception as e:
                update_exception = e
                debug_print(f"❌ Original update failed: {type(e).__name__}: {e}")
                debug_print(f"\n🔍 Full traceback:")
                
                # Capture traceback to buffer
                tb_buffer = io.StringIO()
                traceback.print_exc(file=tb_buffer)
                debug_print(tb_buffer.getvalue())
                
                # Last resort: return inputs unchanged
                debug_print(f"🚨 CRITICAL: Returning inputs unchanged as last resort")
                result = key_states, value_states
            
            # Only do comprehensive logging on failures or periodically
            should_log_details = (
                update_exception or 
                not test_success or 
                call_count % 1000 == 0 or  # Log details every 1000 calls
                call_count < 10  # Log first 10 calls for initialization tracking
            )
            
            if should_log_details:
                # Write all debug output to log file
                log_debug(debug_buffer.getvalue())
            
            # Only print summary to console
            if update_exception or not test_success:
                print(f"❌ Cache debug #{call_count}: {'FAILED' if update_exception else 'WARNING'} - see {patch_comprehensive_cache_debugging.log_path}")
            elif call_count % 1000 == 0:  # Print progress every 1000 calls
                print(f"✅ Cache debug #{call_count}: OK - logging to {patch_comprehensive_cache_debugging.log_path}")
                # Also log a summary every 1000 calls
                log_debug(f"=== SUMMARY at call #{call_count} ===")
                log_debug(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
                log_debug(f"Cache state: {'None' if not hasattr(self, 'keys') or self.keys is None else 'Initialized'}")
                log_debug("")
            
            return result
        
        # Apply the patch
        SlidingWindowLayer.update = comprehensive_debug_update
        print(f"✅ Applied comprehensive cache debugging patch with file logging")
        print(f"📝 Logs will be written to: {patch_comprehensive_cache_debugging.log_path}")
        print(f"💡 Set UNSLOTH_CACHE_DEBUG_LOG_DIR env var to change log directory")
        
    except Exception as e:
        print(f"❌ Failed to apply comprehensive cache debugging patch: {e}")
        traceback.print_exc()

# CRITICAL: Execute the patch immediately on module import!
print("="*80)
print("🚨 EXECUTING patch_comprehensive_cache_debugging() NOW!")
print("="*80)
patch_comprehensive_cache_debugging()
print("="*80)
print("✅ patch_comprehensive_cache_debugging() COMPLETED!")
print("="*80)

# Write marker to debug file to confirm patch was loaded
try:
    import os
    os.makedirs('/dbfs/FileStore/payer_ai/NLP/CALLM', exist_ok=True)
    with open('/dbfs/FileStore/payer_ai/NLP/CALLM/cache_debug.txt', 'a') as f:
        import datetime
        f.write(f"\n\n{'='*80}\n")
        f.write(f"🔧 PATCH LOADED at {datetime.datetime.now()}\n")
        f.write(f"comprehensive_cache_debug patch with cache_position fix is ACTIVE\n")
        f.write(f"{'='*80}\n\n")
except Exception as e:
    print(f"Could not write patch marker: {e}")

# Add to temporary patches
TEMPORARY_PATCHES.append(patch_comprehensive_cache_debugging)