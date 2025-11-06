# GRPO Cache Position Overflow Bug Fix

## Problem Summary
We're debugging and fixing a critical cache position overflow issue in GRPO (Group Relative Policy Optimization) training with the GPT-OSS-20B model using unsloth. The bug causes training to crash with IndexError during the backward pass.

## Issue Description
- **Model**: `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (20B parameters, 4-bit quantized, sliding window architecture)
- **Training**: GRPO training crashes during backward pass 
- **Error Pattern**: Cache positions jump from normal values (~63) to massive values (1262+, 1156+ elements)
- **Root Cause**: Sliding window cache using total cache length (3299) instead of actual window size (128)
- **Impact**: Training fails with `IndexError: index_copy_(): Number of indices (1) should be equal to source.size(dim) (1156)`

## Key Technical Details
- **Sliding Window Size**: 128 tokens (correct)
- **Max Cache Length**: 3299 tokens (total capacity, not window size) 
- **Bug**: Code was confusing max_cache_len (3299) with sliding_window (128)
- **Symptom**: cache_position tensors accumulate thousands of elements instead of staying scalar/single-element

## Libraries Being Modified
We're making fixes to **both** unsloth ecosystem libraries, not the training scripts:

### 1. unsloth-zoo (Primary Focus)
- **Repository**: `https://github.com/amrothemich/unsloth-zoo.git`
- **Branch**: `fix-grpo` 
- **Key Files**:
  - `unsloth_zoo/temporary_patches/cache_position_fix.py` - Core cache position overflow fix
  - `unsloth_zoo/temporary_patches/dtype_fix.py` - Mixed precision dtype consistency 
  - `unsloth_zoo/temporary_patches/minimal_cache_fix.py` - Cache device handling
  - `unsloth_zoo/temporary_patches/__init__.py` - Patch registration

### 2. unsloth (Secondary)
- **Repository**: `https://github.com/amrothemich/unsloth.git` 
- **Branch**: `fix-grpo`
- **Focus**: Supporting fixes if needed, but most work is in unsloth-zoo

## Fixes Implemented

### Critical Fix: Sliding Window Size Correction
**File**: `unsloth_zoo/temporary_patches/cache_position_fix.py`
```python
# BEFORE (buggy):
self._actual_window_size = max_cache_len  # 3299 - WRONG!

# AFTER (fixed):  
self._actual_window_size = sliding_window if sliding_window is not None else 128  # 128 - CORRECT!
```

### Cache Position Accumulation Prevention
- Detect when cache_position tensors have multiple elements (e.g., 1156 positions)
- Force back to single-element tensors before corruption spreads
- Add position wrapping for sliding windows (position 128 → 0, 129 → 1, etc.)

### Dtype Consistency Fixes
- Resolve "BFloat16 and Float" dtype mismatches in mixed precision training
- Ensure all matrix operations use consistent dtypes

## Testing Strategy
- **Modal Environment**: Using Modal for GPU testing since it matches production environment
- **Reproduction**: Create minimal reproducers that trigger the exact cache corruption
- **Validation**: Test forward pass, backward pass, evaluation, and full GRPO training
- **Conservative Approach**: Extensive logging to detect when repairs trigger

## Current Status
- ✅ **Root cause identified**: Window size confusion (3299 vs 128)
- ✅ **Core fix implemented**: Using correct sliding window size
- ✅ **Cache corruption detection**: Comprehensive logging and repair mechanisms  
- ✅ **Fixes committed**: Available on `fix-grpo` branch
- 🔄 **Validation in progress**: Testing that fixes don't introduce silent corruption

## Installation
```bash
# Install both libraries with fixes
pip install 'git+https://github.com/amrothemich/unsloth.git@fix-grpo#egg=unsloth[cu124-torch260]' \
            'git+https://github.com/amrothemich/unsloth-zoo.git@fix-grpo#egg=unsloth-zoo'
```

## Key Commits
- `e2e6b74`: Prevent cache position accumulation at source + dtype fixes
- `c8f0afc`: Critical fix: Use sliding_window (128) instead of max_cache_len (3299)
- `41a01fa`: Fix UnboundLocalError in cache position overflow handling

## Environment Requirements
- **GPU**: A100-40GB (for testing with Modal)
- **CUDA**: 12.4+  
- **PyTorch**: 2.6.0+cu124
- **Transformers**: 4.56.2
- **Model**: Must use GPT-OSS-20B (not Llama variants)

## Debugging Tools
- **Environment Variable**: `UNSLOTH_CACHE_DEBUG_LOG_DIR` - Enable cache debugging logs
- **Abort Mode**: `UNSLOTH_ABORT_ON_CACHE_CORRUPTION=1` - Crash instead of silent repair
- **Log Analysis**: Check for patterns like "🚨 SlidingWindow: CORRUPTED" in logs

## Next Steps
1. **Validation**: Run comprehensive tests to ensure fixes don't break model correctness
2. **Performance**: Verify training/inference performance is not degraded
3. **Production**: Deploy to production training once validation passes
4. **Upstream**: Consider contributing fixes back to upstream unsloth if appropriate

## Files for Testing
- `modal_cache_reproducer.py` - Full GRPO reproduction test
- `modal_test_cache_fix.py` - Simplified cache fix validation
- `validate_cache_understanding.py` - Systematic validation suite

This is a **library-level fix** that patches the underlying cache system behavior, not a training script modification.

## Development Guidelines

### Always Commit and Push Changes
**IMPORTANT**: When making any changes to unsloth or unsloth-zoo repositories, always commit and push the changes immediately after implementation. Do not leave changes uncommitted.

- Use descriptive commit messages that explain the fix/feature
- Include the 🤖 Generated with [Claude Code](https://claude.ai/code) signature
- Push to the `fix-grpo` branch for both repositories
- This ensures all fixes are preserved and available for testing/deployment