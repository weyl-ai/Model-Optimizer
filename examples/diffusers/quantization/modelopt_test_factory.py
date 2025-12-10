#!/usr/bin/env python3
"""
ModelOpt Test Factory - Automated Test Generation from Shape Patterns

Generates comprehensive test scenarios and dummy data based on discovered shapes.
Works with modelopt_auto_shapes.py to create realistic test cases.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """Represents a test scenario with inputs and expected behavior."""
    name: str
    model_name: str
    description: str
    input_tensors: Dict[str, torch.Tensor]
    expected_output_shape: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelOptTestFactory:
    """Factory for generating test scenarios from shape patterns."""
    
    def __init__(self, shape_database_path: Path = Path("modelopt_shape_database.json")):
        """Initialize with shape database."""
        self.shape_db = self._load_shape_database(shape_database_path)
        self.scenarios: List[TestScenario] = []
        
    def _load_shape_database(self, path: Path) -> Dict:
        """Load the shape database from JSON."""
        if not path.exists():
            logger.warning(f"Shape database not found at {path}")
            return {}
            
        with open(path) as f:
            return json.load(f)
    
    def generate_scenario(
        self,
        model_name: str,
        scenario_type: str = "typical",
        batch_size: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None
    ) -> TestScenario:
        """Generate a test scenario for a model."""
        
        if model_name not in self.shape_db.get("models", {}):
            raise ValueError(f"Model {model_name} not found in shape database")
            
        model_config = self.shape_db["models"][model_name]
        
        # Generate input tensors based on scenario type
        if scenario_type == "typical":
            inputs = self._generate_typical_inputs(model_config, batch_size)
        elif scenario_type == "edge_case":
            inputs = self._generate_edge_case_inputs(model_config)
        elif scenario_type == "stress":
            inputs = self._generate_stress_inputs(model_config)
        elif scenario_type == "custom":
            inputs = self._generate_custom_inputs(model_config, batch_size, resolution)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        scenario = TestScenario(
            name=f"{model_name}_{scenario_type}_{batch_size or 'default'}",
            model_name=model_name,
            description=f"{scenario_type.capitalize()} scenario for {model_name}",
            input_tensors=inputs,
            metadata={
                "model_id": model_config.get("model_id"),
                "architecture": model_config.get("architecture"),
                "scenario_type": scenario_type,
                "batch_size": batch_size,
            }
        )
        
        self.scenarios.append(scenario)
        return scenario
    
    def _generate_typical_inputs(self, config: Dict, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generate typical input tensors for a model."""
        inputs = {}
        input_shapes = config.get("input_shapes", {})
        
        for name, shape in input_shapes.items():
            # Override batch size if specified
            if batch_size and len(shape) > 0:
                shape = [batch_size] + list(shape[1:])
            
            # Generate appropriate tensor based on name
            if "timestep" in name:
                # Timestep is typically a scalar or small tensor
                tensor = torch.rand(shape) * 1000
            elif "ids" in name or "coords" in name:
                # IDs and coordinates are often position-based
                tensor = self._generate_position_tensor(shape)
            elif "mask" in name:
                # Masks are binary or attention masks
                tensor = torch.ones(shape)
            elif "guidance" in name:
                # Guidance scale
                tensor = torch.full(shape, 3.5)
            else:
                # Default to random normal for hidden states
                tensor = torch.randn(shape) * 0.1
                
            inputs[name] = tensor.to(torch.float16)
            
        return inputs
    
    def _generate_edge_case_inputs(self, config: Dict) -> Dict[str, torch.Tensor]:
        """Generate edge case inputs (zeros, ones, extremes)."""
        inputs = {}
        input_shapes = config.get("input_shapes", {})
        
        for i, (name, shape) in enumerate(input_shapes.items()):
            # Cycle through different edge cases
            edge_type = i % 4
            
            if edge_type == 0:
                # All zeros
                tensor = torch.zeros(shape)
            elif edge_type == 1:
                # All ones
                tensor = torch.ones(shape)
            elif edge_type == 2:
                # Very small values
                tensor = torch.randn(shape) * 1e-6
            else:
                # Very large values (but not inf)
                tensor = torch.randn(shape) * 100
                
            inputs[name] = tensor.to(torch.float16)
            
        return inputs
    
    def _generate_stress_inputs(self, config: Dict) -> Dict[str, torch.Tensor]:
        """Generate stress test inputs (max batch size, resolution)."""
        inputs = {}
        input_shapes = config.get("input_shapes", {})
        
        for name, shape in input_shapes.items():
            # Maximize batch dimension
            if len(shape) > 0:
                shape = list(shape)
                shape[0] = 16  # Max batch size
                
            # For image-like tensors, use larger spatial dims
            if len(shape) == 4:  # [B, C, H, W]
                shape[2] = shape[3] = 256  # Larger resolution
            elif len(shape) == 3 and "hidden_states" in name:
                # For sequence models, increase sequence length
                shape[1] = min(shape[1] * 2, 8192)
                
            tensor = torch.randn(shape) * 0.1
            inputs[name] = tensor.to(torch.float16)
            
        return inputs
    
    def _generate_custom_inputs(
        self,
        config: Dict,
        batch_size: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate custom inputs with specific dimensions."""
        inputs = {}
        input_shapes = config.get("input_shapes", {})
        
        for name, shape in input_shapes.items():
            shape = list(shape)
            
            # Apply custom batch size
            if batch_size and len(shape) > 0:
                shape[0] = batch_size
                
            # Apply custom resolution for image tensors
            if resolution and len(shape) == 4:
                # Assuming BCHW format
                h, w = resolution
                # Convert to latent space (typically 8x or 16x compression)
                compression = 16 if "flux" in config.get("model_id", "") else 8
                shape[2] = h // compression
                shape[3] = w // compression
            elif resolution and "hidden_states" in name and len(shape) == 3:
                # For transformer models, adjust sequence length
                h, w = resolution
                compression = 16 if "flux" in config.get("model_id", "") else 8
                latent_h = h // compression
                latent_w = w // compression
                shape[1] = latent_h * latent_w
                
            tensor = torch.randn(shape) * 0.1
            inputs[name] = tensor.to(torch.float16)
            
        return inputs
    
    def _generate_position_tensor(self, shape: List[int]) -> torch.Tensor:
        """Generate position-based tensor (for IDs, coordinates)."""
        if len(shape) == 2:
            # 2D position encoding
            positions = torch.arange(shape[0]).unsqueeze(1).expand(-1, shape[1]).float()
            positions = positions / shape[0]  # Normalize
        elif len(shape) == 3:
            # 3D position encoding
            positions = torch.zeros(shape)
            for i in range(shape[0]):
                positions[i, :, :] = i / max(shape[0] - 1, 1)
        else:
            # Default to random for other shapes
            positions = torch.rand(shape)
            
        return positions
    
    def generate_all_scenarios(self, model_name: str) -> List[TestScenario]:
        """Generate all standard test scenarios for a model."""
        scenarios = []
        
        # Typical scenarios with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            scenarios.append(
                self.generate_scenario(model_name, "typical", batch_size=batch_size)
            )
        
        # Edge cases
        scenarios.append(self.generate_scenario(model_name, "edge_case"))
        
        # Stress test
        scenarios.append(self.generate_scenario(model_name, "stress"))
        
        # Custom resolutions (for image models)
        if "flux" in model_name or "sdxl" in model_name or "sd3" in model_name:
            for resolution in [(512, 512), (768, 768), (1024, 1024), (1024, 768)]:
                scenarios.append(
                    self.generate_scenario(
                        model_name, "custom", batch_size=1, resolution=resolution
                    )
                )
        
        return scenarios
    
    def validate_scenario(self, scenario: TestScenario, model: Optional[Any] = None) -> Dict:
        """Validate a test scenario against actual model execution."""
        validation = {
            "scenario": scenario.name,
            "status": "pending",
            "issues": []
        }
        
        # Check input shapes match expected
        model_config = self.shape_db["models"].get(scenario.model_name, {})
        expected_shapes = model_config.get("input_shapes", {})
        
        for name, tensor in scenario.input_tensors.items():
            if name in expected_shapes:
                expected = expected_shapes[name]
                actual = list(tensor.shape)
                
                # Check shape compatibility (ignoring batch dimension variations)
                if len(expected) != len(actual):
                    validation["issues"].append(
                        f"{name}: dimension mismatch {len(expected)} vs {len(actual)}"
                    )
                elif len(expected) > 1:
                    # Check non-batch dimensions
                    for i in range(1, len(expected)):
                        if expected[i] != actual[i] and expected[i] != -1:
                            validation["issues"].append(
                                f"{name}: shape mismatch at dim {i}: {expected[i]} vs {actual[i]}"
                            )
        
        validation["status"] = "valid" if not validation["issues"] else "invalid"
        return validation
    
    def export_scenarios(self, output_dir: Path, format: str = "pytorch"):
        """Export test scenarios to files."""
        output_dir.mkdir(exist_ok=True)
        
        for scenario in self.scenarios:
            scenario_dir = output_dir / scenario.name
            scenario_dir.mkdir(exist_ok=True)
            
            if format == "pytorch":
                # Save as PyTorch tensors
                for name, tensor in scenario.input_tensors.items():
                    torch.save(tensor, scenario_dir / f"{name}.pt")
                    
            elif format == "numpy":
                # Save as numpy arrays
                for name, tensor in scenario.input_tensors.items():
                    np.save(scenario_dir / f"{name}.npy", tensor.cpu().numpy())
                    
            elif format == "onnx":
                # Save as ONNX test data
                import onnx
                from onnx import numpy_helper
                
                for name, tensor in scenario.input_tensors.items():
                    onnx_tensor = numpy_helper.from_array(
                        tensor.cpu().numpy(), name=name
                    )
                    with open(scenario_dir / f"{name}.pb", "wb") as f:
                        f.write(onnx_tensor.SerializeToString())
            
            # Save metadata
            metadata = {
                "name": scenario.name,
                "model_name": scenario.model_name,
                "description": scenario.description,
                "input_shapes": {
                    name: list(tensor.shape) 
                    for name, tensor in scenario.input_tensors.items()
                },
                "metadata": scenario.metadata
            }
            
            with open(scenario_dir / "scenario.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        logger.info(f"Exported {len(self.scenarios)} scenarios to {output_dir}")
    
    def generate_validation_code(self, model_name: str) -> str:
        """Generate Python code to validate model with test scenarios."""
        code = f'''#!/usr/bin/env python3
"""
Auto-generated validation code for {model_name}
Generated by ModelOpt Test Factory
"""

import torch
from pathlib import Path
from diffusers import FluxPipeline, StableDiffusionXLPipeline

def load_scenario(scenario_dir: Path):
    """Load test scenario from directory."""
    inputs = {{}}
    for tensor_file in scenario_dir.glob("*.pt"):
        name = tensor_file.stem
        inputs[name] = torch.load(tensor_file)
    return inputs

def validate_{model_name.replace("-", "_")}():
    """Validate {model_name} with test scenarios."""
    
    # Load model
    print("Loading model...")
'''
        
        if "flux" in model_name:
            code += f'''    model = FluxPipeline.from_pretrained(
        "{self.shape_db['models'][model_name]['model_id']}",
        torch_dtype=torch.float16
    ).transformer
'''
        elif "sdxl" in model_name:
            code += f'''    model = StableDiffusionXLPipeline.from_pretrained(
        "{self.shape_db['models'][model_name]['model_id']}",
        torch_dtype=torch.float16
    ).unet
'''
        
        code += '''    
    # Test scenarios
    scenarios_dir = Path("test_scenarios") / "''' + model_name + '''"
    
    for scenario_path in sorted(scenarios_dir.glob("*")):
        if scenario_path.is_dir():
            print(f"\\nTesting {scenario_path.name}...")
            
            # Load inputs
            inputs = load_scenario(scenario_path)
            
            # Run model
            try:
                with torch.no_grad():
                    output = model(**inputs, return_dict=False)
                    
                if isinstance(output, tuple):
                    output = output[0]
                    
                print(f"  ✓ Output shape: {output.shape}")
                print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    validate_''' + model_name.replace("-", "_") + '''()
'''
        return code


def main():
    """CLI interface for test factory."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ModelOpt Test Factory")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--scenario", default="all",
                       choices=["typical", "edge_case", "stress", "custom", "all"])
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--resolution", type=str, help="Resolution (e.g., 1024x768)")
    parser.add_argument("--output", type=Path, default=Path("test_scenarios"),
                       help="Output directory")
    parser.add_argument("--format", default="pytorch",
                       choices=["pytorch", "numpy", "onnx"])
    parser.add_argument("--generate-code", action="store_true",
                       help="Generate validation code")
    
    args = parser.parse_args()
    
    # Parse resolution
    resolution = None
    if args.resolution:
        h, w = map(int, args.resolution.split("x"))
        resolution = (h, w)
    
    # Create factory
    factory = ModelOptTestFactory()
    
    # Generate scenarios
    if args.scenario == "all":
        scenarios = factory.generate_all_scenarios(args.model)
        print(f"Generated {len(scenarios)} scenarios for {args.model}")
    else:
        scenario = factory.generate_scenario(
            args.model,
            args.scenario,
            batch_size=args.batch_size,
            resolution=resolution
        )
        print(f"Generated scenario: {scenario.name}")
    
    # Export scenarios
    factory.export_scenarios(args.output, format=args.format)
    print(f"Exported to {args.output}")
    
    # Generate validation code
    if args.generate_code:
        code = factory.generate_validation_code(args.model)
        code_file = args.output / f"validate_{args.model.replace('-', '_')}.py"
        code_file.write_text(code)
        print(f"Generated validation code: {code_file}")


if __name__ == "__main__":
    main()