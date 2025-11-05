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

def patch_comprehensive_cache_debugging():
    """
    Comprehensive cache debugging to track memory corruption progression
    """
    try:
        from transformers.cache_utils import SlidingWindowLayer
        import torch
        import sys
        
        # Global counter and state tracking
        if not hasattr(patch_comprehensive_cache_debugging, 'call_count'):
            patch_comprehensive_cache_debugging.call_count = 0
            patch_comprehensive_cache_debugging.failure_count = 0
            patch_comprehensive_cache_debugging.last_successful_state = {}
            patch_comprehensive_cache_debugging.device_history = []
        
        original_update = SlidingWindowLayer.update
        
        def comprehensive_debug_update(self, key_states, value_states, cache_kwargs):
            call_count = patch_comprehensive_cache_debugging.call_count
            patch_comprehensive_cache_debugging.call_count += 1
            
            print(f"\n{'='*80}")
            print(f"🔍 CACHE DEBUG #{call_count} - SlidingWindowLayer.update")
            print(f"{'='*80}")
            
            # Get calling context
            stack = traceback.extract_stack()
            calling_context = []
            for frame in reversed(stack[:-1]):  # Skip current frame
                if 'transformers' in frame.filename or 'unsloth' in frame.filename:
                    calling_context.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
                    if len(calling_context) >= 5:  # Show last 5 relevant frames
                        break
            
            print("📍 Call Stack:")
            for ctx in calling_context:
                print(ctx)
            
            # Detailed state inspection
            print(f"\n📊 State Information:")
            print(f"  self.device: {getattr(self, 'device', 'MISSING')}")
            print(f"  self.device type: {type(getattr(self, 'device', None))}")
            print(f"  self.device id: {id(getattr(self, 'device', None))}")
            print(f"  key_states.device: {key_states.device}")
            print(f"  key_states.shape: {key_states.shape}")
            print(f"  key_states.dtype: {key_states.dtype}")
            print(f"  key_states.is_contiguous: {key_states.is_contiguous()}")
            
            # Memory state
            if torch.cuda.is_available():
                print(f"\n💾 GPU Memory State:")
                print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                
                # Check for memory pressure
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = total_memory - torch.cuda.memory_allocated()
                print(f"  Free Memory: {free_memory / 1e9:.2f} GB ({free_memory/total_memory*100:.1f}%)")
            
            # Cache state inspection
            if hasattr(self, 'keys') and self.keys is not None:
                print(f"\n📦 Cache State:")
                print(f"  Keys shape: {self.keys.shape}")
                print(f"  Keys device: {self.keys.device}")
                print(f"  Keys dtype: {self.keys.dtype}")
                print(f"  Keys contiguous: {self.keys.is_contiguous()}")
                print(f"  Keys data_ptr: {self.keys.data_ptr()}")
                
                # Check for NaN or Inf
                try:
                    has_nan = torch.isnan(self.keys).any().item()
                    has_inf = torch.isinf(self.keys).any().item()
                    print(f"  Keys has NaN: {has_nan}")
                    print(f"  Keys has Inf: {has_inf}")
                except Exception as e:
                    print(f"  ❌ Cannot check NaN/Inf: {e}")
            
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
                print(f"\n🔧 CRITICAL: self.device missing! Setting to key_states.device")
                self.device = key_states.device
                device_ok = False
            elif self.device is None:
                print(f"\n🔧 CRITICAL: self.device is None! Setting to key_states.device")
                self.device = key_states.device
                device_ok = False
            elif isinstance(self.device, int):
                print(f"\n🔧 Converting integer device {self.device} to cuda:{self.device}")
                self.device = f"cuda:{self.device}"
                device_ok = False
            
            # Test the problematic operation
            print(f"\n🧪 Testing index tensor creation...")
            test_success = False
            try:
                test_index = torch.tensor([-1], dtype=int, device=self.device)
                print(f"  ✅ Test index creation successful")
                test_success = True
                
                # Additional memory access test
                if hasattr(self, 'keys') and self.keys is not None:
                    print(f"  🧪 Testing cache memory access...")
                    _ = self.keys[0, 0, 0, 0].item()  # Try to access first element
                    print(f"  ✅ Cache memory access successful")
                    
                    # Test roll operation
                    print(f"  🧪 Testing roll operation...")
                    _ = self.keys.roll(-1, dims=-2)
                    print(f"  ✅ Roll operation successful")
                    
            except Exception as e:
                print(f"  ❌ Test failed: {type(e).__name__}: {e}")
                patch_comprehensive_cache_debugging.failure_count += 1
                
                if "illegal memory access" in str(e):
                    print(f"\n🚨 MEMORY CORRUPTION DETECTED!")
                    print(f"  This is failure #{patch_comprehensive_cache_debugging.failure_count}")
                    print(f"  Total calls before failure: {call_count}")
                    
                    # Compare with last successful state
                    if patch_comprehensive_cache_debugging.last_successful_state:
                        print(f"\n📊 Comparison with last successful state:")
                        last = patch_comprehensive_cache_debugging.last_successful_state
                        print(f"  Last device: {last.get('device')} -> Current: {self.device}")
                        print(f"  Last allocated: {last.get('allocated', 0):.2f} GB -> Current: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                        print(f"  Memory increase: {(torch.cuda.memory_allocated() - last.get('allocated', 0)) / 1e9:.2f} GB")
                    
                    # Show recent device history
                    print(f"\n📜 Recent device history:")
                    for entry in patch_comprehensive_cache_debugging.device_history[-5:]:
                        print(f"  Call #{entry['call']}: device={entry['device']}, key_device={entry['key_device']}")
                    
                    # Emergency fix attempt
                    print(f"\n🚑 Attempting emergency fixes...")
                    
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"  ✅ Forced garbage collection")
                    
                    # Reset device
                    self.device = key_states.device
                    print(f"  ✅ Reset device to {self.device}")
                    
                    # Try again
                    try:
                        test_index = torch.tensor([-1], dtype=int, device=self.device)
                        print(f"  ✅ Emergency fix successful!")
                        test_success = True
                    except Exception as e2:
                        print(f"  ❌ Emergency fix failed: {e2}")
            
            # Store successful state
            if test_success and device_ok:
                patch_comprehensive_cache_debugging.last_successful_state = {
                    'call': call_count,
                    'device': self.device,
                    'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'cache_shape': self.keys.shape if hasattr(self, 'keys') and self.keys is not None else None
                }
            
            # Call original update
            print(f"\n🔄 Calling original update method...")
            try:
                result = original_update(self, key_states, value_states, cache_kwargs)
                print(f"✅ Original update successful")
                return result
            except Exception as e:
                print(f"❌ Original update failed: {type(e).__name__}: {e}")
                print(f"\n🔍 Full traceback:")
                traceback.print_exc()
                
                # Last resort: return inputs unchanged
                print(f"🚨 CRITICAL: Returning inputs unchanged as last resort")
                return key_states, value_states
        
        # Apply the patch
        SlidingWindowLayer.update = comprehensive_debug_update
        print("✅ Applied comprehensive cache debugging patch with memory corruption tracking")
        
    except Exception as e:
        print(f"❌ Failed to apply comprehensive cache debugging patch: {e}")
        traceback.print_exc()

# Add to temporary patches
TEMPORARY_PATCHES.append(patch_comprehensive_cache_debugging)