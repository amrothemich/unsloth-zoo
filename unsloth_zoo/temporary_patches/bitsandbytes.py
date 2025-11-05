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

import torch
import torch.nn as nn
import inspect
import importlib
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from .common import TEMPORARY_PATCHES, torch_compile
from .utils import (
    patch_function,
    process_output_options,
    process_return,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    Unpack,
    _get_unique_storage_name,
)
from textwrap import dedent
import re


def patch_bitsandbytes_linear4bit_forward():
    # Fixes torch.compile complaining about multiple things
    print("🔧 CRITICAL: COMPREHENSIVE PATCH VERSION 2025-11-05-20:15 - Attempting to patch ALL bitsandbytes Linear4bit forward...")
    try:
        import bitsandbytes
        import bitsandbytes.nn
        import bitsandbytes.nn.modules
        fix_4bit_weight_quant_state_from_module = bitsandbytes.nn.modules.fix_4bit_weight_quant_state_from_module
        print("🔧 CRITICAL: Successfully imported bitsandbytes modules")
    except Exception as e:
        print(f"🔧 CRITICAL: Failed to import bitsandbytes: {e}")
        return raise_error("bitsandbytes.Linear4bit", e)

    def forward(self, x: torch.Tensor):
        fix_4bit_weight_quant_state_from_module(self)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        
        # ** Errors out in torch.compile so remove it
        # if self.bias is not None and self.bias.dtype != x.dtype:
        #     self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        
        # FIX: The original code that fails:
        # weight = self.weight.data.t() if self.weight.dim() == 2 else self.weight.data
        # The .data attribute access causes dynamo compilation to fail
        print(f"🔧 USING COMPREHENSIVE BITSANDBYTES FORWARD VERSION 2025-11-05-20:15 DYNAMO-SAFE")
        
        # Use the weight directly without .data attribute access
        if self.weight.dim() == 2:
            weight = self.weight.t()
        else:
            weight = self.weight

        return bitsandbytes.matmul_4bit(x, weight, bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)

    # Patch ALL possible locations where Linear4bit might be used
    locations_to_patch = [
        ("bitsandbytes.nn.modules.Linear4bit", bitsandbytes.nn.modules.Linear4bit),
        ("bitsandbytes.nn.Linear4bit", bitsandbytes.nn.Linear4bit),
        ("bitsandbytes.Linear4bit", getattr(bitsandbytes, 'Linear4bit', None)),
    ]
    
    patched_count = 0
    for location_name, location_class in locations_to_patch:
        if location_class is not None:
            try:
                patch_function(location_class, "forward", forward)
                print(f"🔧 CRITICAL: Successfully patched {location_name}")
                patched_count += 1
            except Exception as e:
                print(f"🔧 CRITICAL: Failed to patch {location_name}: {e}")
    
    print(f"🔧 CRITICAL: Patched {patched_count} Linear4bit locations")
pass
print(f"🔧 CRITICAL: Adding bitsandbytes patch to TEMPORARY_PATCHES list (current length: {len(TEMPORARY_PATCHES)})")
TEMPORARY_PATCHES.append(patch_bitsandbytes_linear4bit_forward)
print(f"🔧 CRITICAL: bitsandbytes patch added, new length: {len(TEMPORARY_PATCHES)}")
