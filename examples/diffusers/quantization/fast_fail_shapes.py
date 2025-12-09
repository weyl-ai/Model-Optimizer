#!/usr/bin/env python3

"""
fast_fail_shapes.py - Validate (FLUX1.dev) DiT input tensor names/shapes against reality
"""

import logging

import torch
from diffusers import FluxPipeline
from diffusers.models.transformers import FluxTransformer2DModel


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose logging

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create custom formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(console_handler)

    # Optionally reduce noise from other libraries
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logger


def inspect_forward_signature(model_id: str = "black-forest-labs/FLUX.1-schnell"):
    """Dump the transformer's forward() signature and a traced call."""

    logger = setup_logging(verbose=True)  # TODO[b7r6]: argument...
    logger.info(f"Loading {model_id}...")

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    transformer: FluxTransformer2DModel = pipe.transformer

    # 1. Inspect forward signature
    import inspect

    sig = inspect.signature(transformer.forward)
    logger.info("[modelopt] [diffusers] forward signature capture")
    for name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else "?"
        default = param.default if param.default != inspect.Parameter.empty else "required"
        logger.info(f"  {name}: {annotation} = {default}")

    # 2. Build dummy inputs matching 1024x1024 generation
    # FLUX latent space is 16x compression, so 1024x1024 -> 64x64 latent -> 4096 tokens
    B = 1
    H, W = 1024, 1024
    latent_h, latent_w = H // 16, W // 16  # 64x64
    N = latent_h * latent_w  # 4096 image tokens
    T = 512  # text sequence length (max_sequence_length default)
    C = transformer.config.in_channels  # latent channels (usually 16 for FLUX)
    hidden_size = transformer.config.joint_attention_dim  # 4096 for FLUX

    logger.info(f"\n=== Derived Dimensions ===")
    logger.info(f"  B={B}, H={H}, W={W}")
    logger.info(f"  latent: {latent_h}x{latent_w} = {N} tokens")
    logger.info(f"  text tokens T={T}")
    logger.info(f"  in_channels C={C}")
    logger.info(f"  hidden_size={hidden_size}")

    # 3. Create dummy tensors - names from FluxTransformer2DModel.forward()
    device = "cuda"
    dtype = torch.bfloat16

    dummy_inputs = {
        "hidden_states": torch.randn(B, N, C, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(B, T, hidden_size, device=device, dtype=dtype),
        "timestep": torch.tensor([0.5], device=device, dtype=dtype),
        "img_ids": torch.zeros(B, N, 3, device=device, dtype=dtype),
        "txt_ids": torch.zeros(B, T, 3, device=device, dtype=dtype),
        "guidance": torch.tensor([3.5], device=device, dtype=dtype),
    }

    logger.info(f"\n=== Dummy Input Shapes ===")
    for name, tensor in dummy_inputs.items():
        logger.info(f"  {name}: {tuple(tensor.shape)} ({tensor.dtype})")

    # 4. Trace a forward pass to confirm it runs
    logger.info(f"\n=== Test Forward Pass ===")
    try:
        with torch.no_grad():
            out = transformer(**dummy_inputs)
        logger.info(
            f"  SUCCESS: output shape = {tuple(out[0].shape) if isinstance(out, tuple) else tuple(out.shape)}"
        )
    except Exception as e:
        logger.info(f"  FAILED: {e}")
        logger.info("  Trying to diagnose required inputs...")

        # Try minimal call to see what's actually needed
        try:
            with torch.no_grad():
                out = transformer(
                    hidden_states=dummy_inputs["hidden_states"],
                    encoder_hidden_states=dummy_inputs["encoder_hidden_states"],
                    timestep=dummy_inputs["timestep"],
                    img_ids=dummy_inputs["img_ids"],
                    txt_ids=dummy_inputs["txt_ids"],
                )
            logger.info(f"  Minimal call (no guidance) worked: {tuple(out[0].shape)}")
        except Exception as e2:
            logger.info(f"  Minimal also failed: {e2}")

    # 5. Export to ONNX and read back input names
    logger.info(f"\n=== ONNX Export Test ===")
    try:
        import tempfile
        import onnx

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        # Prep inputs as tuple for export
        example_inputs = (
            dummy_inputs["hidden_states"],
            dummy_inputs["encoder_hidden_states"],
            dummy_inputs["timestep"],
            dummy_inputs["img_ids"],
            dummy_inputs["txt_ids"],
        )

        input_names = ["hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids"]

        torch.onnx.export(
            transformer,
            example_inputs,
            onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes={
                "hidden_states": {0: "B", 1: "N"},
                "encoder_hidden_states": {0: "B", 1: "T"},
                "timestep": {0: "B"},
                "img_ids": {0: "B", 1: "N"},
                "txt_ids": {0: "B", 1: "T"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

        model = onnx.load(onnx_path)
        logger.info("  ONNX inputs:")
        for inp in model.graph.input:
            dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info(f"    {inp.name}: {dims}")

        logger.info("  ONNX outputs:")
        for out in model.graph.output:
            dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info(f"    {out.name}: {dims}")

        logger.info(f"\n  Wrote: {onnx_path}")

    except Exception as e:
        logger.info(f"  ONNX export failed: {e}")

    # 6. Logger.Info suggested dynamic_axes dict
    logger.info(f"Suggested _FLUX_DYNAMIC_AXES")
    logger.info("""
_FLUX_DYNAMIC_AXES = {
    "hidden_states": {0: "B", 1: "N"},           # [B, H*W/256, C]
    "encoder_hidden_states": {0: "B", 1: "T"},   # [B, seq_len, hidden]
    "timestep": {0: "B"},
    "img_ids": {0: "B", 1: "N"},                 # [B, H*W/256, 3]
    "txt_ids": {0: "B", 1: "T"},                 # [B, seq_len, 3]
}
""")

    # 7. Logger.Info trtexec shapes
    logger.info(f"=== trtexec shape strings ===")
    min_shapes = f"hidden_states:1x{N}x{C},encoder_hidden_states:1x256x{hidden_size},timestep:1,img_ids:1x{N}x3,txt_ids:1x256x3"
    opt_shapes = f"hidden_states:1x{N}x{C},encoder_hidden_states:1x{T}x{hidden_size},timestep:1,img_ids:1x{N}x3,txt_ids:1x{T}x3"
    max_shapes = f"hidden_states:4x{N}x{C},encoder_hidden_states:4x{T}x{hidden_size},timestep:4,img_ids:4x{N}x3,txt_ids:4x{T}x3"

    logger.info(f"  --minShapes={min_shapes}")
    logger.info(f"  --optShapes={opt_shapes}")
    logger.info(f"  --maxShapes={max_shapes}")


if __name__ == "__main__":
    inspect_forward_signature()
