#!/usr/bin/env python3
"""
ModelOpt Automatic Shape Discovery and Configuration System

A unified tool for discovering, analyzing, and generating shape configurations
for NVIDIA Model Optimizer quantization pipelines. Eliminates manual shape annotation.

This module automatically:
1. Discovers tensor shapes from PyTorch models
2. Predicts ONNX export naming and quantization behavior
3. Generates dummy inputs and dynamic axes configurations
4. Creates quantization configs with proper shape alignment
5. Validates configurations against actual model execution
"""

import json
import logging
import inspect
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Supported model architectures
MODEL_CONFIGS = {
    "flux-schnell": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "arch": "FluxTransformer2DModel",
        "latent_channels": 16,
        "hidden_size": 3072,
        "text_hidden": 4096,
        "guidance": False,
    },
    "flux-dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "arch": "FluxTransformer2DModel",
        "latent_channels": 16,
        "hidden_size": 3072,
        "text_hidden": 4096,
        "guidance": True,
    },
    "flux2-dev": {
        "model_id": "black-forest-labs/FLUX.2-dev",
        "arch": "FluxTransformer2DModel",
        "latent_channels": 16,
        "hidden_size": 3072,
        "text_hidden": 4096,
        "guidance": True,
    },
    "sdxl-base": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "arch": "UNet2DConditionModel",
        "latent_channels": 4,
        "cross_attention_dim": 2048,
        "sample_size": 128,
    },
    "sd3-medium": {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "arch": "SD3Transformer2DModel",
        "latent_channels": 16,
        "hidden_size": 1536,
        "pooled_projection_dim": 2048,
    },
}


@dataclass
class TensorShape:
    """Represents a tensor's shape and metadata."""

    name: str
    shape: List[int]
    dtype: str
    dynamic_axes: Dict[int, str] = field(default_factory=dict)
    size_mb: float = 0.0

    def is_aligned(self, alignment: int = 128) -> bool:
        """Check if tensor dimensions are aligned to given boundary."""
        for dim in self.shape:
            if dim % alignment != 0:
                return False
        return True


@dataclass
class ShapeConfig:
    """Complete shape configuration for a model."""

    model_name: str
    model_id: str
    input_shapes: Dict[str, TensorShape]
    output_shapes: Dict[str, TensorShape]
    weight_shapes: Dict[str, TensorShape]
    dynamic_axes: Dict[str, Dict[int, str]]
    quantization_patterns: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelOptShapeAnalyzer:
    """Unified shape analyzer for Model Optimizer pipelines."""

    def __init__(self):
        self.configs: Dict[str, ShapeConfig] = {}
        self._load_existing_configs()

    def _load_existing_configs(self):
        """Load pre-calculated configurations if they exist."""
        config_file = Path("modelopt_shape_database.json")
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)
                for model_name, model_data in data.get("models", {}).items():
                    # Create simplified config from database
                    self.configs[model_name] = self._parse_database_config(model_name, model_data)
                    
    def _parse_database_config(self, model_name: str, data: Dict) -> ShapeConfig:
        """Parse database entry into ShapeConfig."""
        input_shapes = {}
        for name, shape in data.get("input_shapes", {}).items():
            input_shapes[name] = TensorShape(
                name=name,
                shape=shape if isinstance(shape, list) else shape["shape"],
                dtype="float16",
                dynamic_axes=data.get("dynamic_axes", {}).get(name, {})
            )
            
        return ShapeConfig(
            model_name=model_name,
            model_id=data.get("model_id", ""),
            input_shapes=input_shapes,
            output_shapes={},
            weight_shapes={},
            dynamic_axes=data.get("dynamic_axes", {}),
            quantization_patterns={},
            metadata={
                "architecture": data.get("architecture", ""),
                "total_parameters": data.get("total_parameters", 0),
            }
        )

    def _parse_existing_config(self, model_name: str, data: Dict) -> ShapeConfig:
        """Parse existing validation data into ShapeConfig."""
        input_shapes = {}
        weight_shapes = {}

        # Extract weight shapes
        for weight in data.get("quantizable_weights", [])[:10]:  # Sample first 10
            weight_shapes[weight["name"]] = TensorShape(
                name=weight["predicted_onnx_name"],
                shape=weight["shape"],
                dtype="float16",
                size_mb=weight["size_mb"],
            )

        return ShapeConfig(
            model_name=model_name,
            model_id=data.get("model_id", ""),
            input_shapes=input_shapes,
            output_shapes={},
            weight_shapes=weight_shapes,
            dynamic_axes={},
            quantization_patterns={},
            metadata={
                "total_parameters": data.get("total_parameters", 0),
                "quantizable_weights": data.get("quantizable_weights", 0),
                "predicted_issues": data.get("predicted_issues", 0),
            },
        )

    def analyze_from_huggingface(self, model_id: str, subfolder: Optional[str] = None) -> ShapeConfig:
        """Load and analyze a model directly from HuggingFace."""
        logger.info(f"Loading model from HuggingFace: {model_id}")
        
        # Try to detect architecture and load model
        model, arch = self._load_huggingface_model(model_id, subfolder)
        
        # Create a name from model_id
        model_name = model_id.split("/")[-1].lower()
        
        # Detect configuration from model
        model_cfg = self._detect_model_config(model, arch)
        model_cfg["model_id"] = model_id
        
        logger.info(f"Detected architecture: {arch}")
        logger.info(f"Analyzing {model_name}...")
        
        # Analyze weights
        weight_shapes = self._analyze_weights(model)
        
        # Generate input shapes based on detected architecture
        input_shapes = self._generate_input_shapes(model, model_cfg)
        
        # If we didn't get shapes, use captured shapes from tracing
        if not input_shapes and hasattr(self, "captured_shapes") and self.captured_shapes:
            logger.info("Using captured shapes from forward tracing")
            for method_name, shape_info in self.captured_shapes.items():
                if "forward" in method_name:
                    for param_name, shape in shape_info.items():
                        if not param_name.startswith("arg_"):
                            input_shapes[param_name] = TensorShape(
                                name=param_name,
                                shape=shape,
                                dtype="float16",
                                dynamic_axes={0: "batch_size"} if len(shape) > 0 else {}
                            )
        
        # Validate we got shapes
        if not input_shapes:
            logger.warning(f"WARNING: No input shapes captured for {model_name}")
            logger.warning(f"Model params: {list(inspect.signature(model.forward).parameters.keys())[:10] if hasattr(model, 'forward') else 'N/A'}")
        
        # Generate dynamic axes
        dynamic_axes = self._generate_dynamic_axes(model_name, model_cfg)
        
        # Extract quantization patterns
        quant_patterns = self._extract_quantization_patterns(model, weight_shapes)
        
        config = ShapeConfig(
            model_name=model_name,
            model_id=model_id,
            input_shapes=input_shapes,
            output_shapes={},
            weight_shapes=weight_shapes,
            dynamic_axes=dynamic_axes,
            quantization_patterns=quant_patterns,
            metadata={
                "arch": arch,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "subfolder": subfolder,
                "captured_shapes": len(self.captured_shapes) if hasattr(self, "captured_shapes") else 0,
            },
        )
        
        self.configs[model_name] = config
        return config
    
    def _load_huggingface_model(self, model_id: str, subfolder: Optional[str] = None):
        """Load model from HuggingFace with architecture detection."""
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import hf_hub_download
        import functools
        
        # First, try to read config.json from the repo
        config_data = {}
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_path) as f:
                config_data = json.load(f)
            logger.info(f"Loaded config.json with keys: {list(config_data.keys())[:10]}")
        except Exception as e:
            logger.debug(f"Could not load config.json: {e}")
            
        # Load the pipeline
        logger.info(f"Loading pipeline with DiffusionPipeline.from_pretrained({model_id})")
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Detect which component to extract
        pipeline_class = type(pipeline).__name__
        logger.info(f"Loaded pipeline type: {pipeline_class}")
        
        # Extract the model component
        model = None
        if hasattr(pipeline, 'transformer'):
            model = pipeline.transformer
            arch = type(model).__name__
        elif hasattr(pipeline, 'unet'):
            model = pipeline.unet
            arch = type(model).__name__
        elif hasattr(pipeline, 'dit'):
            model = pipeline.dit
            arch = type(model).__name__
        else:
            for attr_name in ['model', 'backbone', 'denoiser', 'net']:
                if hasattr(pipeline, attr_name):
                    model = getattr(pipeline, attr_name)
                    arch = type(model).__name__
                    break
        
        if model is None:
            raise ValueError(f"Could not find model component in pipeline")
            
        # Wrap forward methods to capture shapes
        self.captured_shapes = {}
        
        def capture_wrapper(name, original_func):
            @functools.wraps(original_func)
            def wrapper(*args, **kwargs):
                # Capture input shapes
                input_info = {}
                for i, arg in enumerate(args[1:]):  # Skip self
                    if torch.is_tensor(arg):
                        input_info[f"arg_{i}"] = list(arg.shape)
                for key, val in kwargs.items():
                    if torch.is_tensor(val):
                        input_info[key] = list(val.shape)
                
                if input_info:
                    self.captured_shapes[name] = input_info
                    logger.debug(f"Captured shapes for {name}: {input_info}")
                    
                return original_func(*args, **kwargs)
            return wrapper
        
        # Wrap forward and __call__ methods
        if hasattr(model, 'forward'):
            model.forward = capture_wrapper('forward', model.forward)
        if hasattr(model, '__call__'):
            model.__call__ = capture_wrapper('__call__', model.__call__)
            
        # Also wrap sub-module forwards for better tracing
        for name, module in model.named_modules():
            if hasattr(module, 'forward') and name:  # Skip root
                module.forward = capture_wrapper(f"{name}.forward", module.forward)
        
        # Store config data in model for later use
        model._loaded_config = config_data
        
        # Clean up pipeline but keep model
        del pipeline
        torch.cuda.empty_cache()
        
        logger.info(f"Extracted {arch} model with config keys: {list(config_data.keys())[:5]}")
        return model, arch
    
    def _detect_model_config(self, model: nn.Module, arch: str) -> Dict:
        """Auto-detect model configuration from loaded model."""
        config = {"arch": arch}
        
        # First, check the loaded config.json
        if hasattr(model, "_loaded_config"):
            loaded_cfg = model._loaded_config
            logger.info(f"Using loaded config.json data")
            
            # Common config keys mapping
            config_mappings = {
                "in_channels": "latent_channels",
                "hidden_size": "hidden_size",
                "joint_attention_dim": "text_hidden",
                "cross_attention_dim": "cross_attention_dim",
                "sample_size": "sample_size",
                "pooled_projection_dim": "pooled_projection_dim",
                "guidance_embeds": "guidance",
                "num_layers": "num_layers",
                "num_attention_heads": "num_attention_heads",
                "patch_size": "patch_size",
                "out_channels": "out_channels",
            }
            
            for json_key, config_key in config_mappings.items():
                if json_key in loaded_cfg:
                    config[config_key] = loaded_cfg[json_key]
                    logger.debug(f"Found {json_key}: {loaded_cfg[json_key]}")
        
        # Then check model.config attribute
        if hasattr(model, "config"):
            model_config = model.config
            
            # Extract relevant dimensions
            if hasattr(model_config, "in_channels"):
                config["latent_channels"] = model_config.in_channels
            if hasattr(model_config, "hidden_size"):
                config["hidden_size"] = model_config.hidden_size
            if hasattr(model_config, "joint_attention_dim"):
                config["text_hidden"] = model_config.joint_attention_dim
            if hasattr(model_config, "cross_attention_dim"):
                config["cross_attention_dim"] = model_config.cross_attention_dim
            if hasattr(model_config, "sample_size"):
                config["sample_size"] = model_config.sample_size
            if hasattr(model_config, "pooled_projection_dim"):
                config["pooled_projection_dim"] = model_config.pooled_projection_dim
            if hasattr(model_config, "guidance_embeds"):
                config["guidance"] = model_config.guidance_embeds
        
        # Try to run a dummy forward pass to capture actual shapes
        if not hasattr(self, "captured_shapes") or not self.captured_shapes:
            self._trace_forward_pass(model, config)
        
        # Use captured shapes to refine config
        if hasattr(self, "captured_shapes") and self.captured_shapes:
            logger.info(f"Refining config with captured shapes from {len(self.captured_shapes)} calls")
            # Analyze captured shapes to infer dimensions
            for method_name, shape_info in self.captured_shapes.items():
                if "forward" in method_name:
                    for param_name, shape in shape_info.items():
                        logger.debug(f"  {param_name}: {shape}")
                        # Infer dimensions from common patterns
                        if len(shape) == 3:  # [batch, seq, hidden]
                            if shape[-1] > 1000:  # Likely hidden dimension
                                config.setdefault("detected_hidden_size", shape[-1])
                        elif len(shape) == 4:  # [batch, channels, height, width]
                            if shape[1] < 100:  # Likely channels
                                config.setdefault("detected_channels", shape[1])
                
        # Fallback to architecture-based defaults
        if arch == "FluxTransformer2DModel":
            config.setdefault("latent_channels", 16)
            config.setdefault("hidden_size", 3072)
            config.setdefault("text_hidden", 4096)
        elif arch == "UNet2DConditionModel":
            config.setdefault("latent_channels", 4)
            config.setdefault("cross_attention_dim", 2048)
            config.setdefault("sample_size", 128)
        elif arch == "SD3Transformer2DModel":
            config.setdefault("latent_channels", 16)
            config.setdefault("hidden_size", 1536)
            config.setdefault("pooled_projection_dim", 2048)
            
        return config
    
    def _trace_forward_pass(self, model: nn.Module, config: Dict):
        """Run a dummy forward pass to capture tensor shapes."""
        import torch
        
        # Create dummy inputs based on what we know
        dummy_inputs = {}
        
        # Check forward signature
        if hasattr(model, "forward"):
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            
            # Remove self and special params
            params = [p for p in params if p not in ["self", "return_dict", "kwargs"]]
            
            logger.info(f"Model expects parameters: {params}")
            
            # Create appropriate dummy tensors based on parameter names
            for param in params[:10]:  # Check more params
                if param == "x" or "hidden" in param or "latent" in param:
                    # For Z-Image, 'x' is the latent input
                    dummy_inputs[param] = torch.randn(1, config.get("latent_channels", 4), 64, 64,
                                                     dtype=torch.float16)
                elif param == "t" or "time" in param:
                    dummy_inputs[param] = torch.tensor([0.5], dtype=torch.float16)
                elif param == "cap_feats" or "encoder" in param or "text" in param or "caption" in param:
                    # Text/caption features
                    dummy_inputs[param] = torch.randn(1, 77, 
                                                     config.get("text_hidden", 2048),
                                                     dtype=torch.float16)
                elif param == "patch_size":
                    dummy_inputs[param] = 2  # Common patch size
                elif param == "f_patch_size":
                    dummy_inputs[param] = 2  # Frame patch size
                elif "pooled" in param:
                    dummy_inputs[param] = torch.randn(1, 768, dtype=torch.float16)
                elif "img_ids" in param or "ids" in param:
                    dummy_inputs[param] = torch.randn(4096, 3, dtype=torch.float16)
                elif "guidance" in param:
                    dummy_inputs[param] = torch.tensor([3.5], dtype=torch.float16)
                    
            # Try to run forward
            if dummy_inputs:
                try:
                    logger.info(f"Running dummy forward with inputs: {list(dummy_inputs.keys())}")
                    with torch.no_grad():
                        model.eval()
                        _ = model(**dummy_inputs, return_dict=False)
                    logger.info(f"Successfully traced forward pass")
                    logger.info(f"Captured shapes: {self.captured_shapes if hasattr(self, 'captured_shapes') else 'None'}")
                except Exception as e:
                    logger.info(f"Dummy forward failed (adjusting inputs): {str(e)[:200]}")
                    
                    # Try simpler inputs for unknown models
                    if "x" in params:
                        dummy_inputs = {
                            "x": torch.randn(1, 4, 64, 64, dtype=torch.float16),
                            "t": torch.tensor([0.5], dtype=torch.float16)
                        }
                        if "cap_feats" in params:
                            dummy_inputs["cap_feats"] = torch.randn(1, 77, 2048, dtype=torch.float16)
                        
                        try:
                            with torch.no_grad():
                                model.eval()
                                _ = model(**dummy_inputs, return_dict=False)
                            logger.info(f"Successfully traced with adjusted inputs")
                        except Exception as e2:
                            logger.warning(f"Could not trace forward: {str(e2)[:100]}")

    def analyze_model(self, model: nn.Module, model_name: str) -> ShapeConfig:
        """Analyze a PyTorch model to extract shape configuration."""
        if model_name in self.configs:
            logger.info(f"Using cached configuration for {model_name}")
            return self.configs[model_name]

        logger.info(f"Analyzing {model_name}...")

        # Get model metadata
        model_cfg = MODEL_CONFIGS.get(model_name, {})

        # Analyze weights
        weight_shapes = self._analyze_weights(model)

        # Generate input shapes
        input_shapes = self._generate_input_shapes(model, model_cfg)

        # Generate dynamic axes
        dynamic_axes = self._generate_dynamic_axes(model_name, model_cfg)

        # Extract quantization patterns
        quant_patterns = self._extract_quantization_patterns(model, weight_shapes)

        config = ShapeConfig(
            model_name=model_name,
            model_id=model_cfg.get("model_id", ""),
            input_shapes=input_shapes,
            output_shapes={},
            weight_shapes=weight_shapes,
            dynamic_axes=dynamic_axes,
            quantization_patterns=quant_patterns,
            metadata={
                "arch": model_cfg.get("arch", ""),
                "total_parameters": sum(p.numel() for p in model.parameters()),
            },
        )

        self.configs[model_name] = config
        return config

    def _analyze_weights(self, model: nn.Module) -> Dict[str, TensorShape]:
        """Extract weight shapes from model."""
        weight_shapes = {}

        for name, param in model.named_parameters():
            if "weight" in name:
                weight_shapes[name] = TensorShape(
                    name=self._predict_onnx_name(name),
                    shape=list(param.shape),
                    dtype=str(param.dtype),
                    size_mb=param.numel() * param.element_size() / (1024**2),
                )

        return weight_shapes

    def _predict_onnx_name(self, pytorch_name: str) -> str:
        """Predict ONNX node name from PyTorch parameter name."""
        # Common transformations
        transforms = [
            (".", "/"),
            ("_weight", "/weight"),
            ("_bias", "/bias"),
            ("transformer_blocks", "/transformer_blocks"),
            ("norm1/linear", "/norm1/linear"),
            ("to_q", "/to_q"),
            ("to_k", "/to_k"),
            ("to_v", "/to_v"),
        ]

        onnx_name = pytorch_name
        for old, new in transforms:
            onnx_name = onnx_name.replace(old, new)

        if not onnx_name.startswith("/"):
            onnx_name = "/" + onnx_name

        return onnx_name

    def _generate_input_shapes(self, model: nn.Module, cfg: Dict) -> Dict[str, TensorShape]:
        """Generate input tensor shapes for model."""
        shapes = {}
        arch = cfg.get("arch", "")

        if arch == "FluxTransformer2DModel" or "flux" in arch.lower():
            shapes = self._generate_flux_inputs(cfg)
        elif arch == "UNet2DConditionModel":
            shapes = self._generate_sdxl_inputs(cfg)
        elif arch == "SD3Transformer2DModel":
            shapes = self._generate_sd3_inputs(cfg)
        elif "qwen" in arch.lower() or "dit" in arch.lower():
            # Generic DiT/Qwen input shapes - similar to FLUX
            shapes = self._generate_dit_inputs(cfg, model)
        else:
            # Try to auto-detect from model's forward signature
            shapes = self._auto_detect_inputs(model, cfg)

        return shapes
    
    def _generate_dit_inputs(self, cfg: Dict, model: nn.Module) -> Dict[str, TensorShape]:
        """Generate input shapes for generic DiT models (including Qwen)."""
        # Default DiT shapes similar to FLUX
        latent_channels = cfg.get("latent_channels", 16)
        hidden_size = cfg.get("hidden_size", cfg.get("text_hidden", 4096))
        
        # Try to detect actual dimensions from model
        if hasattr(model, "config"):
            if hasattr(model.config, "in_channels"):
                latent_channels = model.config.in_channels
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
        
        img_dim = 4096  # Default for 1024x1024
        text_len = 512
        
        shapes = {
            "hidden_states": TensorShape(
                name="hidden_states",
                shape=[1, img_dim, latent_channels],
                dtype="float16",
                dynamic_axes={0: "batch_size", 1: "latent_dim"},
            ),
            "timestep": TensorShape(
                name="timestep", 
                shape=[1], 
                dtype="float32", 
                dynamic_axes={0: "batch_size"}
            ),
        }
        
        # Add text conditioning if model expects it
        if hasattr(model, "forward"):
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            
            if "encoder_hidden_states" in params or "text_embeds" in params:
                shapes["encoder_hidden_states"] = TensorShape(
                    name="encoder_hidden_states",
                    shape=[1, text_len, hidden_size],
                    dtype="float16",
                    dynamic_axes={0: "batch_size"},
                )
            
            if "pooled_projections" in params:
                shapes["pooled_projections"] = TensorShape(
                    name="pooled_projections",
                    shape=[1, 768],  # Common pooled size
                    dtype="float16",
                    dynamic_axes={0: "batch_size"},
                )
        
        return shapes
    
    def _auto_detect_inputs(self, model: nn.Module, cfg: Dict) -> Dict[str, TensorShape]:
        """Auto-detect input shapes from model's forward signature."""
        shapes = {}
        
        if hasattr(model, "forward"):
            import inspect
            sig = inspect.signature(model.forward)
            
            # Common parameter patterns
            for param_name in sig.parameters:
                if param_name in ["self", "return_dict"]:
                    continue
                    
                # Make educated guesses based on parameter names
                if "hidden" in param_name or "latent" in param_name:
                    shapes[param_name] = TensorShape(
                        name=param_name,
                        shape=[1, 4096, 16],  # Default latent shape
                        dtype="float16",
                        dynamic_axes={0: "batch_size", 1: "sequence_length"},
                    )
                elif "time" in param_name:
                    shapes[param_name] = TensorShape(
                        name=param_name,
                        shape=[1],
                        dtype="float32",
                        dynamic_axes={0: "batch_size"},
                    )
                elif "text" in param_name or "encoder" in param_name:
                    shapes[param_name] = TensorShape(
                        name=param_name,
                        shape=[1, 512, 4096],  # Default text shape
                        dtype="float16",
                        dynamic_axes={0: "batch_size", 1: "sequence_length"},
                    )
        
        return shapes

    def _generate_flux_inputs(self, cfg: Dict) -> Dict[str, TensorShape]:
        """Generate FLUX model input shapes."""
        img_dim = 4096  # 1024x1024 -> 64x64 latent -> 4096 tokens
        text_len = 512

        shapes = {
            "hidden_states": TensorShape(
                name="hidden_states",
                shape=[1, img_dim, cfg["latent_channels"]],
                dtype="float16",
                dynamic_axes={0: "batch_size", 1: "latent_dim"},
            ),
            "encoder_hidden_states": TensorShape(
                name="encoder_hidden_states",
                shape=[1, text_len, cfg["text_hidden"]],
                dtype="float16",
                dynamic_axes={0: "batch_size"},
            ),
            "pooled_projections": TensorShape(
                name="pooled_projections",
                shape=[1, 768],
                dtype="float16",
                dynamic_axes={0: "batch_size"},
            ),
            "timestep": TensorShape(
                name="timestep", shape=[1], dtype="float32", dynamic_axes={0: "batch_size"}
            ),
            "img_ids": TensorShape(
                name="img_ids", shape=[img_dim, 3], dtype="float32", dynamic_axes={0: "latent_dim"}
            ),
        }

        if cfg.get("guidance"):
            shapes["guidance"] = TensorShape(
                name="guidance", shape=[1], dtype="float32", dynamic_axes={0: "batch_size"}
            )

        return shapes

    def _generate_sdxl_inputs(self, cfg: Dict) -> Dict[str, TensorShape]:
        """Generate SDXL model input shapes."""
        sample_size = cfg.get("sample_size", 128)

        return {
            "sample": TensorShape(
                name="sample",
                shape=[2, cfg["latent_channels"], sample_size, sample_size],
                dtype="float16",
                dynamic_axes={0: "batch_size", 2: "height", 3: "width"},
            ),
            "timestep": TensorShape(
                name="timestep", shape=[1], dtype="int64", dynamic_axes={0: "steps"}
            ),
            "encoder_hidden_states": TensorShape(
                name="encoder_hidden_states",
                shape=[2, 77, cfg["cross_attention_dim"]],
                dtype="float16",
                dynamic_axes={0: "batch_size", 1: "sequence_length"},
            ),
            "text_embeds": TensorShape(
                name="text_embeds", shape=[2, 1280], dtype="float16", dynamic_axes={0: "batch_size"}
            ),
            "time_ids": TensorShape(
                name="time_ids", shape=[2, 6], dtype="float16", dynamic_axes={0: "batch_size"}
            ),
        }

    def _generate_sd3_inputs(self, cfg: Dict) -> Dict[str, TensorShape]:
        """Generate SD3 model input shapes."""
        return {
            "hidden_states": TensorShape(
                name="hidden_states",
                shape=[2, cfg["latent_channels"], 128, 128],
                dtype="float16",
                dynamic_axes={0: "batch_size", 2: "height", 3: "width"},
            ),
            "timestep": TensorShape(
                name="timestep", shape=[2], dtype="float32", dynamic_axes={0: "steps"}
            ),
            "encoder_hidden_states": TensorShape(
                name="encoder_hidden_states",
                shape=[2, 333, cfg["hidden_size"]],
                dtype="float16",
                dynamic_axes={0: "batch_size", 1: "sequence_length"},
            ),
            "pooled_projections": TensorShape(
                name="pooled_projections",
                shape=[2, cfg["pooled_projection_dim"]],
                dtype="float16",
                dynamic_axes={0: "batch_size"},
            ),
        }

    def _generate_dynamic_axes(self, model_name: str, cfg: Dict) -> Dict[str, Dict[int, str]]:
        """Generate dynamic axes configuration."""
        # This maps directly to MODEL_ID_TO_DYNAMIC_AXES in export.py
        if "flux" in model_name:
            return {
                "hidden_states": {0: "batch_size", 1: "latent_dim"},
                "encoder_hidden_states": {0: "batch_size"},
                "pooled_projections": {0: "batch_size"},
                "timestep": {0: "batch_size"},
                "img_ids": {0: "latent_dim"},
                "latent": {0: "batch_size"},
            }
        elif "sdxl" in model_name:
            return {
                "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
                "timestep": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
                "text_embeds": {0: "batch_size"},
                "time_ids": {0: "batch_size"},
                "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            }
        elif "sd3" in model_name:
            return {
                "hidden_states": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
                "timestep": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
                "pooled_projections": {0: "batch_size"},
                "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            }
        return {}

    def _extract_quantization_patterns(self, model: nn.Module, weight_shapes: Dict) -> Dict:
        """Extract patterns for quantization configuration."""
        patterns = {
            "quantizable_layers": [],
            "skip_patterns": [],
            "fp4_eligible": [],
            "fp8_eligible": [],
        }

        # Identify quantizable layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_name = f"{name}.weight"
                if weight_name in weight_shapes:
                    shape = weight_shapes[weight_name]

                    # Check FP4 eligibility (aligned to 128)
                    if shape.is_aligned(128):
                        patterns["fp4_eligible"].append(name)
                    elif shape.is_aligned(16):
                        patterns["fp8_eligible"].append(name)

                    # Check if should skip (embedding, normalization, etc)
                    skip_keywords = ["embed", "norm", "pos_", "rope", "time_"]
                    if any(kw in name.lower() for kw in skip_keywords):
                        patterns["skip_patterns"].append(name)
                    else:
                        patterns["quantizable_layers"].append(name)

        return patterns

    def generate_config_py(self, model_name: str, output_file: Optional[Path] = None) -> str:
        """Generate config.py replacement with discovered shapes."""
        if model_name not in self.configs:
            raise ValueError(f"No configuration found for {model_name}")

        config = self.configs[model_name]

        # Generate quantization config based on patterns
        code = self._generate_config_code(config)

        if output_file:
            output_file.write_text(code)
            logger.info(f"Generated config saved to {output_file}")

        return code

    def _generate_config_code(self, config: ShapeConfig) -> str:
        """Generate Python code for quantization config."""
        patterns = config.quantization_patterns

        code = f"""# Auto-generated quantization config for {config.model_name}
# Generated by ModelOpt Automatic Shape Discovery System

import torch.nn as nn
from calib.plugin_calib import PercentileCalibrator

# Model: {config.model_id}
# Total parameters: {config.metadata.get("total_parameters", 0):,}
# Quantizable layers: {len(patterns.get("quantizable_layers", []))}
# FP4 eligible: {len(patterns.get("fp4_eligible", []))}
# FP8 eligible: {len(patterns.get("fp8_eligible", []))}

NVFP4_E2M1 = {{
    "num_bits": (2, 1),
    "block_sizes": {{-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
    "enable": True,
}}

FP8_E4M3 = {{"num_bits": (4, 3), "axis": None}}

# Auto-discovered quantization config
AUTO_NVFP4_CONFIG = {{
    "quant_cfg": {{
"""

        # Add quantizable patterns
        for layer in patterns.get("fp4_eligible", [])[:20]:  # First 20 as example
            code += f'        "*{layer}*weight_quantizer": NVFP4_E2M1,\n'
            code += f'        "*{layer}*input_quantizer": NVFP4_E2M1,\n'

        # Add skip patterns
        for layer in patterns.get("skip_patterns", [])[:10]:
            code += f'        "*{layer}*": {{"enable": False}},\n'

        code += '''        "*output_quantizer": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

def get_auto_config(model_name: str):
    """Get auto-discovered config for model."""
    return AUTO_NVFP4_CONFIG
'''
        return code

    def generate_export_shapes(self, model_name: str) -> Dict:
        """Generate dummy inputs and dynamic shapes for ONNX export."""
        if model_name not in self.configs:
            raise ValueError(f"No configuration found for {model_name}")

        config = self.configs[model_name]

        # Create shape dictionary (NOT actual tensors, just shapes!)
        input_shapes = {}
        for name, shape_info in config.input_shapes.items():
            input_shapes[name] = {
                "shape": shape_info.shape,
                "dtype": shape_info.dtype
            }

        return {
            "input_shapes": input_shapes,
            "dynamic_axes": config.dynamic_axes,
            "input_names": list(config.input_shapes.keys()),
            "output_names": ["latent"] if "flux" in model_name else ["sample"],
        }

    def validate_against_reality(self, model_name: str, actual_shapes: Dict) -> Dict:
        """Validate predicted shapes against actual execution."""
        if model_name not in self.configs:
            return {"status": "error", "message": "No config found"}

        config = self.configs[model_name]
        mismatches = []

        for name, actual_shape in actual_shapes.items():
            if name in config.input_shapes:
                predicted = config.input_shapes[name].shape
                if predicted != actual_shape:
                    mismatches.append(
                        {"tensor": name, "predicted": predicted, "actual": actual_shape}
                    )

        return {
            "status": "success" if not mismatches else "mismatch",
            "mismatches": mismatches,
            "accuracy": 1.0 - len(mismatches) / len(actual_shapes) if actual_shapes else 1.0,
        }

    def export_all_configs(self, output_dir: Path, generate_code: bool = False):
        """Export all discovered configurations."""
        output_dir.mkdir(exist_ok=True)

        # Save JSON database
        all_configs = {}
        for name, config in self.configs.items():
            all_configs[name] = {
                "model_id": config.model_id,
                "input_shapes": {k: asdict(v) for k, v in config.input_shapes.items()},
                "weight_shapes": {k: asdict(v) for k, v in config.weight_shapes.items()},
                "dynamic_axes": config.dynamic_axes,
                "quantization_patterns": config.quantization_patterns,
                "metadata": config.metadata,
            }

        json_file = output_dir / "modelopt_shape_database.json"
        with open(json_file, "w") as f:
            json.dump(all_configs, f, indent=2)
        logger.info(f"Saved shape database to {json_file}")

        if generate_code:
            # Generate config.py replacement
            config_file = output_dir / "auto_config.py"
            with open(config_file, "w") as f:
                f.write("# Auto-generated configs for all models\n\n")
                for name in self.configs:
                    f.write(f"\n# === {name} ===\n")
                    f.write(self.generate_config_py(name))
            logger.info(f"Generated config code at {config_file}")

            # Generate export.py helpers
            export_file = output_dir / "auto_export_shapes.py"
            with open(export_file, "w") as f:
                f.write("# Auto-generated export shapes for all models\n\n")
                f.write("EXPORT_CONFIGS = {\n")
                for name in self.configs:
                    shapes = self.generate_export_shapes(name)
                    f.write(f'    "{name}": {shapes},\n')
                f.write("}\n")
            logger.info(f"Generated export shapes at {export_file}")


def main():
    """CLI interface for shape discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="ModelOpt Automatic Shape Discovery")
    parser.add_argument("--model", help="Model to analyze (predefined name or HuggingFace model ID)")
    parser.add_argument("--hf-model", help="HuggingFace model ID to analyze")
    parser.add_argument("--subfolder", help="Subfolder for model component (e.g., 'transformer')")
    parser.add_argument("--all", action="store_true", help="Analyze all predefined models")
    parser.add_argument("--generate", action="store_true", help="Generate config replacements")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modelopt_configs"),
        help="Output directory for generated configs",
    )
    parser.add_argument("--validate", type=str, help="Validate shapes against execution trace JSON")
    parser.add_argument("--save-db", action="store_true", help="Save analyzed config to database")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analyzer = ModelOptShapeAnalyzer()

    if args.validate:
        # Load and validate
        with open(args.validate) as f:
            actual = json.load(f)
        for model_name, shapes in actual.items():
            result = analyzer.validate_against_reality(model_name, shapes)
            print(f"{model_name}: {result['status']} (accuracy: {result['accuracy']:.1%})")

    elif args.hf_model or (args.model and "/" in args.model):
        # Analyze HuggingFace model
        model_id = args.hf_model or args.model
        print(f"\n🤗 Analyzing HuggingFace model: {model_id}")
        
        try:
            config = analyzer.analyze_from_huggingface(model_id, args.subfolder)
            
            print(f"\n✅ Successfully analyzed {model_id}")
            print(f"Architecture: {config.metadata['arch']}")
            print(f"Total parameters: {config.metadata['total_parameters']:,}")
            print(f"Input tensors: {list(config.input_shapes.keys())}")
            
            if args.generate:
                # Generate configs
                args.output.mkdir(exist_ok=True)
                
                # Generate config.py
                config_code = analyzer.generate_config_py(config.model_name)
                config_file = args.output / f"{config.model_name}_config.py"
                config_file.write_text(config_code)
                print(f"\n📝 Generated config: {config_file}")
                
                # Generate export shapes
                export_shapes = analyzer.generate_export_shapes(config.model_name)
                export_file = args.output / f"{config.model_name}_shapes.json"
                with open(export_file, "w") as f:
                    json.dump(export_shapes, f, indent=2)
                print(f"📝 Generated shapes: {export_file}")
            
            if args.save_db:
                # Update database
                db_path = Path("modelopt_shape_database.json")
                if db_path.exists():
                    with open(db_path) as f:
                        db = json.load(f)
                else:
                    db = {"description": "ModelOpt shape database", "version": "1.0.0", "models": {}}
                
                db["models"][config.model_name] = {
                    "model_id": config.model_id,
                    "architecture": config.metadata['arch'],
                    "total_parameters": config.metadata['total_parameters'],
                    "input_shapes": {k: v.shape for k, v in config.input_shapes.items()},
                    "dynamic_axes": config.dynamic_axes,
                }
                
                with open(db_path, "w") as f:
                    json.dump(db, f, indent=2)
                print(f"💾 Saved to database: {db_path}")
                
        except Exception as e:
            print(f"\n❌ Failed to analyze {model_id}: {e}")
            import traceback
            traceback.print_exc()

    elif args.all:
        # Analyze all predefined models
        for model_name in MODEL_CONFIGS:
            print(f"\nAnalyzing {model_name}...")
            try:
                model_id = MODEL_CONFIGS[model_name]["model_id"]
                config = analyzer.analyze_from_huggingface(model_id)
                print(f"✓ {model_name} analyzed")
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")

        # Export everything
        analyzer.export_all_configs(args.output, generate_code=args.generate)

    elif args.model and args.model in MODEL_CONFIGS:
        # Single predefined model analysis
        print(f"Analyzing predefined model: {args.model}...")
        model_id = MODEL_CONFIGS[args.model]["model_id"]
        
        try:
            config = analyzer.analyze_from_huggingface(model_id)
            
            if args.generate:
                code = analyzer.generate_config_py(args.model)
                print("\n=== Generated Config ===")
                print(code)

                shapes = analyzer.generate_export_shapes(args.model)
                print("\n=== Export Shapes ===")
                print(json.dumps(shapes, indent=2, default=str))
        except Exception as e:
            print(f"Failed: {e}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Analyze a HuggingFace model:")
        print("  python modelopt_auto_shapes.py --hf-model black-forest-labs/FLUX.1-dev --generate")
        print("  python modelopt_auto_shapes.py --model stabilityai/stable-diffusion-xl-base-1.0 --generate")
        print("\n  # Analyze predefined models:")
        print("  python modelopt_auto_shapes.py --model flux-dev --generate")
        print("  python modelopt_auto_shapes.py --all --generate")


if __name__ == "__main__":
    main()
