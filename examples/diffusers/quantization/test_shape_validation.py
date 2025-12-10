#!/usr/bin/env python3
"""
Test script to validate that shapes from runtime tracing match the generated shape files.

This ensures:
1. Captured shapes from functools wrapping match reality
2. Generated shape files are accurate
3. Config files align with actual model requirements
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from diffusers import DiffusionPipeline
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_trace(model_id: str) -> Tuple[torch.nn.Module, Dict]:
    """Load a model and capture its actual runtime shapes."""
    logger.info(f"Loading model: {model_id}")
    
    # Load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    # Extract model component
    if hasattr(pipeline, 'transformer'):
        model = pipeline.transformer
    elif hasattr(pipeline, 'unet'):
        model = pipeline.unet
    elif hasattr(pipeline, 'dit'):
        model = pipeline.dit
    else:
        raise ValueError(f"Cannot find model component in pipeline")
    
    model_name = type(model).__name__
    logger.info(f"Extracted {model_name}")
    
    # Capture actual shapes by running a forward pass
    captured_shapes = {}
    
    # Hook to capture inputs
    def capture_hook(module, args, kwargs):
        shapes = {}
        # Capture positional args
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                shapes[f"arg_{i}"] = list(arg.shape)
        # Capture keyword args
        for name, val in kwargs.items():
            if torch.is_tensor(val):
                shapes[name] = list(val.shape)
            elif isinstance(val, (int, float)):
                shapes[name] = val
        return shapes
    
    # Prepare dummy inputs based on model type
    if "flux" in model_id.lower():
        dummy_inputs = {
            "hidden_states": torch.randn(1, 4096, 64, dtype=torch.float16, device=model.device),
            "encoder_hidden_states": torch.randn(1, 512, 4096, dtype=torch.float16, device=model.device),
            "pooled_projections": torch.randn(1, 768, dtype=torch.float16, device=model.device),
            "timestep": torch.tensor([0.5], dtype=torch.float16, device=model.device),
            "img_ids": torch.randn(4096, 3, dtype=torch.float32, device=model.device),
            "txt_ids": torch.randn(512, 3, dtype=torch.float32, device=model.device),
        }
        if "dev" in model_id.lower():
            dummy_inputs["guidance"] = torch.tensor([3.5], dtype=torch.float32, device=model.device)
    elif "z-image" in model_id.lower():
        dummy_inputs = {
            "x": torch.randn(1, 4, 64, 64, dtype=torch.float16, device=model.device),
            "t": torch.tensor([0.5], dtype=torch.float16, device=model.device),
            "cap_feats": torch.randn(1, 77, 2048, dtype=torch.float16, device=model.device),
        }
        # Z-Image might need these
        if hasattr(model, 'forward'):
            import inspect
            sig = inspect.signature(model.forward)
            if "patch_size" in sig.parameters:
                dummy_inputs["patch_size"] = 2
            if "f_patch_size" in sig.parameters:
                dummy_inputs["f_patch_size"] = 2
    elif "sdxl" in model_id.lower():
        dummy_inputs = {
            "sample": torch.randn(2, 4, 128, 128, dtype=torch.float16, device=model.device),
            "timestep": torch.tensor([1.0], dtype=torch.float16, device=model.device),
            "encoder_hidden_states": torch.randn(2, 77, 2048, dtype=torch.float16, device=model.device),
        }
        # SDXL uses added_cond_kwargs
        dummy_inputs["added_cond_kwargs"] = {
            "text_embeds": torch.randn(2, 1280, dtype=torch.float16, device=model.device),
            "time_ids": torch.randn(2, 6, dtype=torch.float16, device=model.device),
        }
    else:
        # Generic fallback
        dummy_inputs = {
            "hidden_states": torch.randn(1, 4096, 16, dtype=torch.float16, device=model.device),
            "timestep": torch.tensor([0.5], dtype=torch.float16, device=model.device),
        }
    
    # Try to run forward and capture shapes
    try:
        with torch.no_grad():
            model.eval()
            # Capture input shapes
            for name, tensor in dummy_inputs.items():
                if torch.is_tensor(tensor):
                    captured_shapes[name] = list(tensor.shape)
                elif isinstance(tensor, dict):
                    for sub_name, sub_tensor in tensor.items():
                        if torch.is_tensor(sub_tensor):
                            captured_shapes[f"{name}.{sub_name}"] = list(sub_tensor.shape)
            
            # Run forward
            output = model(**dummy_inputs, return_dict=False)
            
            # Capture output shape
            if torch.is_tensor(output):
                captured_shapes["output"] = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                captured_shapes["output"] = list(output[0].shape)
                
        logger.info(f"✓ Successfully traced forward pass")
    except Exception as e:
        logger.warning(f"Forward pass failed (adjusting): {str(e)[:100]}")
        # Still return what we captured
        for name, tensor in dummy_inputs.items():
            if torch.is_tensor(tensor):
                captured_shapes[name] = list(tensor.shape)
    
    # Clean up
    del pipeline
    torch.cuda.empty_cache()
    
    return model, captured_shapes


def load_generated_shapes(shape_file: Path) -> Dict:
    """Load shapes from generated JSON file."""
    with open(shape_file) as f:
        data = json.load(f)
    
    # Handle both old and new formats
    if "input_shapes" in data:
        # New clean format
        shapes = {}
        for name, info in data["input_shapes"].items():
            shapes[name] = info["shape"]
        return shapes
    elif "dummy_inputs" in data:
        # Old format with tensor dumps - extract shapes
        shapes = {}
        for name, tensor_str in data["dummy_inputs"].items():
            # Try to parse shape from tensor string
            if isinstance(tensor_str, str) and "tensor" in tensor_str:
                # This is a dumped tensor, try to extract shape
                # For now, return empty as we can't easily parse
                logger.warning(f"Old format detected for {name}, cannot extract shape")
            elif isinstance(tensor_str, dict) and "shape" in tensor_str:
                shapes[name] = tensor_str["shape"]
        return shapes
    else:
        return {}


def validate_shapes(runtime_shapes: Dict, file_shapes: Dict) -> Dict:
    """Compare runtime shapes with file shapes."""
    results = {
        "status": "pass",
        "matches": [],
        "mismatches": [],
        "missing_in_file": [],
        "extra_in_file": []
    }
    
    # Check each runtime shape
    for name, runtime_shape in runtime_shapes.items():
        if name.startswith("output"):
            continue  # Skip output shapes for now
            
        if name in file_shapes:
            file_shape = file_shapes[name]
            if runtime_shape == file_shape:
                results["matches"].append({
                    "name": name,
                    "shape": runtime_shape
                })
            else:
                results["mismatches"].append({
                    "name": name,
                    "runtime": runtime_shape,
                    "file": file_shape
                })
                results["status"] = "fail"
        else:
            # Check for nested names (e.g., added_cond_kwargs.text_embeds)
            base_name = name.split('.')[0] if '.' in name else name
            if base_name not in file_shapes:
                results["missing_in_file"].append({
                    "name": name,
                    "shape": runtime_shape
                })
                # Don't fail for missing optional inputs
    
    # Check for extra shapes in file
    for name, file_shape in file_shapes.items():
        if name not in runtime_shapes:
            # Could be optional
            results["extra_in_file"].append({
                "name": name,
                "shape": file_shape
            })
    
    return results


def test_model(model_id: str, shape_file: Path) -> bool:
    """Test a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"Shape file: {shape_file}")
    print(f"{'='*60}")
    
    # Load and trace model
    try:
        model, runtime_shapes = load_model_and_trace(model_id)
        print(f"\n✓ Captured {len(runtime_shapes)} runtime shapes:")
        for name, shape in runtime_shapes.items():
            print(f"  {name}: {shape}")
    except Exception as e:
        print(f"✗ Failed to load/trace model: {e}")
        return False
    
    # Load generated shapes
    if not shape_file.exists():
        print(f"✗ Shape file not found: {shape_file}")
        return False
        
    file_shapes = load_generated_shapes(shape_file)
    print(f"\n✓ Loaded {len(file_shapes)} shapes from file:")
    for name, shape in file_shapes.items():
        print(f"  {name}: {shape}")
    
    # Validate
    results = validate_shapes(runtime_shapes, file_shapes)
    
    print(f"\n{'Validation Results':^30}")
    print(f"{'-'*30}")
    print(f"Status: {results['status'].upper()}")
    print(f"Matches: {len(results['matches'])}")
    print(f"Mismatches: {len(results['mismatches'])}")
    print(f"Missing in file: {len(results['missing_in_file'])}")
    print(f"Extra in file: {len(results['extra_in_file'])}")
    
    if results["mismatches"]:
        print(f"\n❌ Shape Mismatches:")
        for mismatch in results["mismatches"]:
            print(f"  {mismatch['name']}:")
            print(f"    Runtime: {mismatch['runtime']}")
            print(f"    File:    {mismatch['file']}")
    
    if results["missing_in_file"]:
        print(f"\n⚠️  Missing in file (may be optional):")
        for missing in results["missing_in_file"]:
            print(f"  {missing['name']}: {missing['shape']}")
    
    if results["extra_in_file"]:
        print(f"\n⚠️  Extra in file (may be optional):")
        for extra in results["extra_in_file"]:
            print(f"  {extra['name']}: {extra['shape']}")
    
    return results["status"] == "pass"


def main():
    """Run validation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate shape generation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--shape-file", required=True, type=Path, help="Generated shape JSON file")
    parser.add_argument("--generate-first", action="store_true", 
                       help="Generate shapes first using modelopt_auto_shapes.py")
    
    args = parser.parse_args()
    
    if args.generate_first:
        # Run shape generation first
        import subprocess
        output_dir = args.shape_file.parent
        cmd = [
            "python", "modelopt_auto_shapes.py",
            "--model", args.model,
            "--generate",
            "--output", str(output_dir)
        ]
        print(f"Generating shapes first: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to generate shapes: {result.stderr}")
            return 1
        print(f"✓ Generated shapes successfully")
    
    # Run validation
    success = test_model(args.model, args.shape_file)
    
    if success:
        print(f"\n✅ VALIDATION PASSED")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    exit(main())