#!/usr/bin/env python3
"""
Quick test to verify our cache position fixes are present
"""

def test_fix_presence():
    """Test that our critical fix is present in the codebase"""
    try:
        # Check if cache position fix is present
        from unsloth_zoo.temporary_patches.cache_position_fix import patch_sliding_window_cache_creation
        import inspect
        
        source = inspect.getsource(patch_sliding_window_cache_creation)
        
        critical_fix_present = "CRITICAL FIX: Use sliding_window (128), NOT max_cache_len (3299)!" in source
        prevention_fix_present = "PREVENTING ACCUMULATION" in source
        
        print("🔍 CACHE POSITION FIX VERIFICATION")
        print("="*50)
        print(f"✅ Critical window size fix present: {critical_fix_present}")
        print(f"✅ Accumulation prevention fix present: {prevention_fix_present}")
        
        if critical_fix_present and prevention_fix_present:
            print("🎉 All critical fixes are present in the codebase!")
            return True
        else:
            print("❌ Some fixes are missing!")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import unsloth_zoo: {e}")
        return False
        
def test_dtype_fix_presence():
    """Test that dtype consistency fixes are present"""
    try:
        from unsloth_zoo.temporary_patches.dtype_fix import patch_dtype_consistency
        import inspect
        
        source = inspect.getsource(patch_dtype_consistency)
        
        dtype_fix_present = "safe_linear_forward" in source
        
        print("\n🔍 DTYPE CONSISTENCY FIX VERIFICATION")
        print("="*50)
        print(f"✅ Dtype consistency fix present: {dtype_fix_present}")
        
        return dtype_fix_present
        
    except ImportError as e:
        print(f"❌ Cannot import dtype fix: {e}")
        return False

if __name__ == "__main__":
    print("🚀 VERIFYING CACHE POSITION OVERFLOW FIXES")
    print("="*60)
    
    cache_fix_ok = test_fix_presence()
    dtype_fix_ok = test_dtype_fix_presence()
    
    print("\n📊 SUMMARY")
    print("="*30)
    
    if cache_fix_ok and dtype_fix_ok:
        print("✅ ALL FIXES VERIFIED - Ready for testing")
        print("\nNext steps:")
        print("1. Test in Modal environment")
        print("2. Validate with GRPO training")
        print("3. Confirm no silent corruption")
    else:
        print("❌ SOME FIXES MISSING - Check installation")
        print("\nTroubleshooting:")
        print("1. Ensure using fix-grpo branch")
        print("2. Check pip installation")
        print("3. Verify unsloth-zoo imports correctly")