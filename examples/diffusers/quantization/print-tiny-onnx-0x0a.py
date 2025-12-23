#!/usr/bin/env python3
"""
Flux.1-dev DiT Forward Pass ONNX Generator with FP4 Quantization Patterns

Architecture (12B params):
  - 19 Double-stream blocks (MMDiT joint attention)
  - 38 Single-stream blocks (standard transformer)
  - hidden_size: 3072, num_heads: 24, head_dim: 128
  - mlp_ratio: 4.0 (mlp_hidden: 12288)
  - T5 context: 4096 dim, CLIP vec: 768 dim
  - VAE latent: 64 channels

Token geometry for 1024x1024:
  - Image: 4096 tokens (128x128 latent / 2x2 packing)
  - Text: 512 tokens (T5 max)
  - Joint: 4608 tokens

FP4 quantization pattern:
  - Weights: FP4E2M1 with per-block FP8E4M3FN scales (block_size=16)
  - Activations: Dynamic FP4 quantization before MatMul
  - Accumulation in FP32, cast to BF16
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import List, Tuple
import argparse

# ═══════════════════════════════════════════════════════════════════════════════
# Flux.1-dev Architecture Constants
# ═══════════════════════════════════════════════════════════════════════════════
HIDDEN = 3072
HEADS = 24
HEAD_DIM = 128  # HIDDEN // HEADS
MLP_HIDDEN = 12288  # HIDDEN * 4
IN_CHANNELS = 64  # VAE latent channels
VEC_DIM = 768  # CLIP pooled embedding
CTX_DIM = 4096  # T5 text embedding
BLOCK_SIZE = 16  # FP4 quantization block size
NUM_DOUBLE = 19  # Double-stream (MMDiT) blocks
NUM_SINGLE = 38  # Single-stream blocks


def make_bf16(name: str, shape: List[int], fill: float = 1.0) -> onnx.TensorProto:
    """Create BF16 tensor with fill value."""
    n = max(1, int(np.prod(shape)))
    data = (np.full(n, fill, np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return helper.make_tensor(name, TensorProto.BFLOAT16, shape, data.tobytes(), raw=True)


def make_shape(name: str, dims: List[int]) -> onnx.TensorProto:
    """Create INT64 shape tensor."""
    return helper.make_tensor(name, TensorProto.INT64, [len(dims)], dims)


class FluxDiTBuilder:
    """Builds Flux.1-dev DiT as ONNX with FP4 quantization patterns."""

    def __init__(
        self, img_seq: int, txt_seq: int, num_double: int = NUM_DOUBLE, num_single: int = NUM_SINGLE
    ):
        self.img_seq = img_seq
        self.txt_seq = txt_seq
        self.joint_seq = img_seq + txt_seq
        self.num_double = num_double
        self.num_single = num_single
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # FP4 Weight and Quantization Primitives
    # ═══════════════════════════════════════════════════════════════════════════

    def fp4_weight(self, name: str, out_dim: int, in_dim: int) -> Tuple[str, str, str]:
        """Create FP4E2M1 weight with FP8 per-block scales."""
        # FP4E2M1: 2 values per byte -> (out_dim, in_dim//2) storage
        w = helper.make_tensor(
            f"{name}_w",
            TensorProto.FLOAT4E2M1,
            [out_dim, in_dim],
            np.zeros((out_dim, in_dim // 2), np.uint8).tobytes(),
            raw=True,
        )
        # Per-block FP8 scales: (out_dim, in_dim // block_size)
        s = helper.make_tensor(
            f"{name}_s",
            TensorProto.FLOAT8E4M3FN,
            [out_dim, in_dim // BLOCK_SIZE],
            np.full((out_dim, in_dim // BLOCK_SIZE), 127, np.uint8).tobytes(),
            raw=True,
        )
        # Global scale (FP32 scalar)
        g = helper.make_tensor(f"{name}_g", TensorProto.FLOAT, [], [1.0])
        self.inits.extend([w, s, g])
        return f"{name}_w", f"{name}_s", f"{name}_g"

    def dequant_weight(self, p: str, w: str, s: str, g: str) -> str:
        """Dequantize FP4 weight: FP4 -> FP32 -> BF16, then transpose for matmul."""
        # Dequant scale: FP8 -> FP32
        self.nodes.append(
            helper.make_node("DequantizeLinear", [s, g], [f"{p}_sf"], name=f"{p}_dqs", domain="trt")
        )
        # Dequant weight: FP4 -> FP32
        self.nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [w, f"{p}_sf"],
                [f"{p}_wf"],
                name=f"{p}_dqw",
                domain="trt",
                axis=-1,
                block_size=BLOCK_SIZE,
            )
        )
        # Cast to BF16
        self.nodes.append(
            helper.make_node(
                "Cast", [f"{p}_wf"], [f"{p}_wb"], name=f"{p}_c", to=TensorProto.BFLOAT16
            )
        )
        # Transpose: (out, in) -> (in, out) for x @ W^T pattern
        self.nodes.append(
            helper.make_node("Transpose", [f"{p}_wb"], [f"{p}_wt"], name=f"{p}_t", perm=[1, 0])
        )
        return f"{p}_wt"

    def dynquant_act(self, p: str, x: str) -> str:
        """Dynamic FP4 quantize activation, then dequant to BF16."""
        # Dynamic quant scale (constant 1.0 for pattern)
        self.nodes.append(
            helper.make_node(
                "Constant",
                [],
                [f"{p}_ds"],
                name=f"{p}_dsc",
                value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
            )
        )
        # TRT_FP4DynamicQuantize: x -> (fp4_data, fp8_scale)
        self.nodes.append(
            helper.make_node(
                "TRT_FP4DynamicQuantize",
                [x, f"{p}_ds"],
                [f"{p}_q", f"{p}_qs"],
                name=f"{p}_dq",
                domain="trt",
                axis=-1,
                block_size=BLOCK_SIZE,
                scale_type=TensorProto.FLOAT8E4M3FN,
            )
        )
        # Dequant scale: FP8 -> FP32
        self.nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{p}_qs", f"{p}_ds"],
                [f"{p}_qsf"],
                name=f"{p}_dqsf",
                domain="trt",
            )
        )
        # Dequant activation: FP4 -> FP32
        self.nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{p}_q", f"{p}_qsf"],
                [f"{p}_xf"],
                name=f"{p}_dqa",
                domain="trt",
                axis=-1,
                block_size=BLOCK_SIZE,
            )
        )
        # Cast to BF16
        self.nodes.append(
            helper.make_node(
                "Cast", [f"{p}_xf"], [f"{p}_xb"], name=f"{p}_cb", to=TensorProto.BFLOAT16
            )
        )
        return f"{p}_xb"

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer Primitives
    # ═══════════════════════════════════════════════════════════════════════════

    def linear(self, p: str, x: str, out_d: int, in_d: int, bias: bool = True) -> str:
        """FP4-quantized linear: y = x @ W^T + b"""
        wn, sn, gn = self.fp4_weight(p, out_d, in_d)
        wt = self.dequant_weight(p, wn, sn, gn)
        xq = self.dynquant_act(f"{p}_a", x)
        self.nodes.append(helper.make_node("MatMul", [xq, wt], [f"{p}_o"], name=f"{p}_mm"))
        if bias:
            self.inits.append(make_bf16(f"{p}_b", [out_d], 0.0))
            self.nodes.append(
                helper.make_node("Add", [f"{p}_o", f"{p}_b"], [f"{p}_ob"], name=f"{p}_ab")
            )
            return f"{p}_ob"
        return f"{p}_o"

    def rms_norm(self, p: str, x: str, dim: int = HIDDEN) -> str:
        """RMSNorm (SimplifiedLayerNormalization in ONNX)."""
        self.inits.append(make_bf16(f"{p}_sc", [dim], 1.0))
        self.nodes.append(
            helper.make_node(
                "SimplifiedLayerNormalization",
                [x, f"{p}_sc"],
                [f"{p}_o"],
                name=p,
                axis=-1,
                epsilon=1e-6,
            )
        )
        return f"{p}_o"

    def layer_norm(self, p: str, x: str, dim: int = HIDDEN) -> str:
        """Standard LayerNorm."""
        self.inits.extend([make_bf16(f"{p}_sc", [dim], 1.0), make_bf16(f"{p}_bi", [dim], 0.0)])
        self.nodes.append(
            helper.make_node(
                "LayerNormalization",
                [x, f"{p}_sc", f"{p}_bi"],
                [f"{p}_o"],
                name=p,
                axis=-1,
                epsilon=1e-5,
            )
        )
        return f"{p}_o"

    def adaln(self, p: str, temb: str, n: int = 6) -> List[str]:
        """AdaLN modulation: SiLU(temb) -> Linear -> Split into n tensors."""
        # SiLU activation
        self.nodes.append(helper.make_node("Sigmoid", [temb], [f"{p}_sig"], name=f"{p}_sig"))
        self.nodes.append(
            helper.make_node("Mul", [temb, f"{p}_sig"], [f"{p}_silu"], name=f"{p}_silu")
        )
        # Project to n * HIDDEN
        lin = self.linear(f"{p}_l", f"{p}_silu", HIDDEN * n, HIDDEN)
        # Split
        outs = [f"{p}_m{i}" for i in range(n)]
        self.nodes.append(
            helper.make_node("Split", [lin], outs, name=f"{p}_sp", axis=-1, num_outputs=n)
        )
        return outs

    def modulate(self, p: str, x: str, shift: str, scale: str) -> str:
        """Apply AdaLN modulation: (1 + scale) * x + shift."""
        self.inits.append(make_bf16(f"{p}_1", [1], 1.0))
        self.nodes.append(helper.make_node("Add", [scale, f"{p}_1"], [f"{p}_sp1"], name=f"{p}_sp1"))
        self.nodes.append(helper.make_node("Mul", [x, f"{p}_sp1"], [f"{p}_sc"], name=f"{p}_sc"))
        self.nodes.append(helper.make_node("Add", [f"{p}_sc", shift], [f"{p}_sh"], name=f"{p}_sh"))
        return f"{p}_sh"

    # ═══════════════════════════════════════════════════════════════════════════
    # Attention Block
    # ═══════════════════════════════════════════════════════════════════════════

    def attention(
        self,
        p: str,
        x: str,
        shift: str,
        scale: str,
        gate: str,
        seq: int,
        kv_seq: int = None,
        ctx: str = None,
    ) -> str:
        """
        Modulated multi-head attention with optional joint context.

        For double-stream blocks: ctx provides cross-stream KV for joint attention.
        For single-stream blocks: ctx=None for self-attention only.
        """
        kv_seq = kv_seq or seq

        # Modulated pre-norm
        n = self.rms_norm(f"{p}_n", x)
        m = self.modulate(f"{p}_mod", n, shift, scale)

        # QKV projection
        qkv = self.linear(f"{p}_qkv", m, HIDDEN * 3, HIDDEN)
        self.nodes.append(
            helper.make_node(
                "Split",
                [qkv],
                [f"{p}_q", f"{p}_k", f"{p}_v"],
                name=f"{p}_qkvs",
                axis=-1,
                num_outputs=3,
            )
        )

        k_name, v_name = f"{p}_k", f"{p}_v"

        # Joint attention: concatenate context KV
        if ctx is not None:
            # Context gets its own KV projection (separate weights per stream in Flux)
            ckv = self.linear(f"{p}_ckv", ctx, HIDDEN * 2, HIDDEN)
            self.nodes.append(
                helper.make_node(
                    "Split", [ckv], [f"{p}_ck", f"{p}_cv"], name=f"{p}_ckvs", axis=-1, num_outputs=2
                )
            )
            # Concat along sequence dimension
            self.nodes.append(
                helper.make_node("Concat", [k_name, f"{p}_ck"], [f"{p}_jk"], name=f"{p}_jk", axis=1)
            )
            self.nodes.append(
                helper.make_node("Concat", [v_name, f"{p}_cv"], [f"{p}_jv"], name=f"{p}_jv", axis=1)
            )
            k_name, v_name = f"{p}_jk", f"{p}_jv"

        # Reshape to multi-head: (B, S, H) -> (B, S, heads, head_dim) -> (B, heads, S, head_dim)
        self.inits.append(make_shape(f"{p}_shq", [1, seq, HEADS, HEAD_DIM]))
        self.inits.append(make_shape(f"{p}_shkv", [1, kv_seq, HEADS, HEAD_DIM]))

        # Q reshape and transpose
        self.nodes.append(
            helper.make_node("Reshape", [f"{p}_q", f"{p}_shq"], [f"{p}_q4"], name=f"{p}_rq")
        )
        self.nodes.append(
            helper.make_node(
                "Transpose", [f"{p}_q4"], [f"{p}_qt"], name=f"{p}_tq", perm=[0, 2, 1, 3]
            )
        )

        # K reshape and transpose
        self.nodes.append(
            helper.make_node("Reshape", [k_name, f"{p}_shkv"], [f"{p}_k4"], name=f"{p}_rk")
        )
        self.nodes.append(
            helper.make_node(
                "Transpose", [f"{p}_k4"], [f"{p}_kt"], name=f"{p}_tk", perm=[0, 2, 1, 3]
            )
        )

        # V reshape and transpose
        self.nodes.append(
            helper.make_node("Reshape", [v_name, f"{p}_shkv"], [f"{p}_v4"], name=f"{p}_rv")
        )
        self.nodes.append(
            helper.make_node(
                "Transpose", [f"{p}_v4"], [f"{p}_vt"], name=f"{p}_tv", perm=[0, 2, 1, 3]
            )
        )

        # FP4 quantize Q, K, V for efficient BMM
        qq = self.dynquant_act(f"{p}_qq", f"{p}_qt")
        kq = self.dynquant_act(f"{p}_kq", f"{p}_kt")
        vq = self.dynquant_act(f"{p}_vq", f"{p}_vt")

        # K^T for attention scores
        self.nodes.append(
            helper.make_node("Transpose", [kq], [f"{p}_kT"], name=f"{p}_tkT", perm=[0, 1, 3, 2])
        )

        # BMM1: Q @ K^T -> (B, heads, seq, kv_seq)
        self.nodes.append(
            helper.make_node("MatMul", [qq, f"{p}_kT"], [f"{p}_sc"], name=f"{p}_bmm1")
        )

        # Scale by 1/sqrt(head_dim)
        self.inits.append(make_bf16(f"{p}_scv", [], 1.0 / np.sqrt(HEAD_DIM)))
        self.nodes.append(
            helper.make_node("Mul", [f"{p}_sc", f"{p}_scv"], [f"{p}_scs"], name=f"{p}_scmul")
        )

        # Softmax over last dim
        self.nodes.append(
            helper.make_node("Softmax", [f"{p}_scs"], [f"{p}_aw"], name=f"{p}_sfm", axis=-1)
        )

        # Quantize attention weights for BMM2
        aq = self.dynquant_act(f"{p}_aq", f"{p}_aw")

        # BMM2: attn @ V -> (B, heads, seq, head_dim)
        self.nodes.append(helper.make_node("MatMul", [aq, vq], [f"{p}_ao"], name=f"{p}_bmm2"))

        # Transpose back: (B, heads, S, head_dim) -> (B, S, heads, head_dim)
        self.nodes.append(
            helper.make_node(
                "Transpose", [f"{p}_ao"], [f"{p}_aot"], name=f"{p}_taob", perm=[0, 2, 1, 3]
            )
        )

        # Reshape to (B, S, H)
        self.inits.append(make_shape(f"{p}_shf", [1, seq, HIDDEN]))
        self.nodes.append(
            helper.make_node("Reshape", [f"{p}_aot", f"{p}_shf"], [f"{p}_aof"], name=f"{p}_raof")
        )

        # Output projection
        proj = self.linear(f"{p}_proj", f"{p}_aof", HIDDEN, HIDDEN)

        # Apply gate
        self.nodes.append(helper.make_node("Mul", [proj, gate], [f"{p}_g"], name=f"{p}_gate"))
        return f"{p}_g"

    # ═══════════════════════════════════════════════════════════════════════════
    # MLP Block
    # ═══════════════════════════════════════════════════════════════════════════

    def mlp(self, p: str, x: str, shift: str, scale: str, gate: str) -> str:
        """Gated MLP with GEGLU activation."""
        # Modulated pre-norm
        n = self.rms_norm(f"{p}_n", x)
        m = self.modulate(f"{p}_mod", n, shift, scale)

        # GEGLU: project to 2x hidden, split, apply GELU to gate half
        up = self.linear(f"{p}_up", m, MLP_HIDDEN * 2, HIDDEN)
        self.nodes.append(
            helper.make_node(
                "Split", [up], [f"{p}_um", f"{p}_ug"], name=f"{p}_ups", axis=-1, num_outputs=2
            )
        )
        self.nodes.append(helper.make_node("Gelu", [f"{p}_ug"], [f"{p}_gelu"], name=f"{p}_gelu"))
        self.nodes.append(
            helper.make_node("Mul", [f"{p}_um", f"{p}_gelu"], [f"{p}_gg"], name=f"{p}_geglu")
        )

        # Down projection
        dn = self.linear(f"{p}_dn", f"{p}_gg", HIDDEN, MLP_HIDDEN)

        # Apply output gate
        self.nodes.append(helper.make_node("Mul", [dn, gate], [f"{p}_o"], name=f"{p}_gate"))
        return f"{p}_o"

    # ═══════════════════════════════════════════════════════════════════════════
    # Transformer Blocks
    # ═══════════════════════════════════════════════════════════════════════════

    def double_block(self, i: int, img: str, txt: str, temb: str) -> Tuple[str, str]:
        """
        Double-stream block (MMDiT style).

        - Separate weights for image and text streams
        - Joint attention: each stream attends to concatenated img+txt tokens
        - Separate MLP for each stream
        - AdaLN-Zero modulation with 6 params per stream
        """
        p = f"d{i}"

        # AdaLN modulation for both streams
        # [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
        im = self.adaln(f"{p}_im", temb, 6)
        tm = self.adaln(f"{p}_tm", temb, 6)

        # Joint attention
        # Image stream attends to img_tokens + txt_tokens
        ia = self.attention(f"{p}_ia", img, im[0], im[1], im[2], self.img_seq, self.joint_seq, txt)
        # Text stream attends to txt_tokens + img_tokens
        ta = self.attention(f"{p}_ta", txt, tm[0], tm[1], tm[2], self.txt_seq, self.joint_seq, img)

        # Attention residual
        self.nodes.append(helper.make_node("Add", [img, ia], [f"{p}_ir1"], name=f"{p}_ir1"))
        self.nodes.append(helper.make_node("Add", [txt, ta], [f"{p}_tr1"], name=f"{p}_tr1"))

        # MLP
        im_mlp = self.mlp(f"{p}_im", f"{p}_ir1", im[3], im[4], im[5])
        tm_mlp = self.mlp(f"{p}_tm", f"{p}_tr1", tm[3], tm[4], tm[5])

        # MLP residual
        self.nodes.append(
            helper.make_node("Add", [f"{p}_ir1", im_mlp], [f"{p}_io"], name=f"{p}_io")
        )
        self.nodes.append(
            helper.make_node("Add", [f"{p}_tr1", tm_mlp], [f"{p}_to"], name=f"{p}_to")
        )

        return f"{p}_io", f"{p}_to"

    def single_block(self, i: int, h: str, temb: str) -> str:
        """
        Single-stream block.

        Standard transformer processing concatenated img+txt tokens.
        AdaLN-Zero with 3 modulation params (fused attention+mlp in Flux).
        """
        p = f"s{i}"

        # AdaLN modulation (3 params)
        m = self.adaln(f"{p}_m", temb, 3)

        # Self-attention over joint sequence
        a = self.attention(f"{p}_a", h, m[0], m[1], m[2], self.joint_seq)

        # Linear projection (single-stream has parallel attention+linear, simplified here)
        lin = self.linear(f"{p}_l", a, HIDDEN, HIDDEN)

        # Residual
        self.nodes.append(helper.make_node("Add", [h, lin], [f"{p}_o"], name=f"{p}_res"))
        return f"{p}_o"

    # ═══════════════════════════════════════════════════════════════════════════
    # Full Model Build
    # ═══════════════════════════════════════════════════════════════════════════

    def build(self) -> onnx.ModelProto:
        """Build complete Flux.1-dev DiT forward pass."""

        # ═══════════════════════════════════════════════════════════════════════
        # Input tensors
        # ═══════════════════════════════════════════════════════════════════════
        inputs = [
            # Image latents after patchify: (B, img_seq, in_channels)
            helper.make_tensor_value_info(
                "img", TensorProto.BFLOAT16, [1, self.img_seq, IN_CHANNELS]
            ),
            # T5 text embeddings: (B, txt_seq, ctx_dim)
            helper.make_tensor_value_info("txt", TensorProto.BFLOAT16, [1, self.txt_seq, CTX_DIM]),
            # CLIP pooled embedding: (B, vec_dim)
            helper.make_tensor_value_info("vec", TensorProto.BFLOAT16, [1, VEC_DIM]),
            # Timestep embedding (pre-computed sinusoidal): (B, hidden)
            helper.make_tensor_value_info("t_emb", TensorProto.BFLOAT16, [1, HIDDEN]),
        ]
        outputs = [
            # Denoised prediction: (B, img_seq, in_channels)
            helper.make_tensor_value_info(
                "out", TensorProto.BFLOAT16, [1, self.img_seq, IN_CHANNELS]
            )
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # Input embeddings
        # ═══════════════════════════════════════════════════════════════════════

        # Project inputs to hidden dimension
        img = self.linear("img_in", "img", HIDDEN, IN_CHANNELS)
        txt = self.linear("txt_in", "txt", HIDDEN, CTX_DIM)
        vec = self.linear("vec_in", "vec", HIDDEN, VEC_DIM)

        # Combine timestep + pooled CLIP for temb
        self.nodes.append(helper.make_node("Add", ["t_emb", vec], ["temb"], name="temb"))

        # ═══════════════════════════════════════════════════════════════════════
        # Double-stream blocks (19)
        # ═══════════════════════════════════════════════════════════════════════
        for i in range(self.num_double):
            img, txt = self.double_block(i, img, txt, "temb")

        # ═══════════════════════════════════════════════════════════════════════
        # Concatenate streams
        # ═══════════════════════════════════════════════════════════════════════
        self.nodes.append(helper.make_node("Concat", [img, txt], ["joint"], name="concat", axis=1))

        # ═══════════════════════════════════════════════════════════════════════
        # Single-stream blocks (38)
        # ═══════════════════════════════════════════════════════════════════════
        h = "joint"
        for i in range(self.num_single):
            h = self.single_block(i, h, "temb")

        # ═══════════════════════════════════════════════════════════════════════
        # Split and output projection
        # ═══════════════════════════════════════════════════════════════════════

        # Split joint back to img/txt
        self.inits.append(
            helper.make_tensor("split_lens", TensorProto.INT64, [2], [self.img_seq, self.txt_seq])
        )
        self.nodes.append(
            helper.make_node("Split", [h, "split_lens"], ["f_img", "f_txt"], name="split", axis=1)
        )

        # Final layer norm
        fn = self.layer_norm("fn", "f_img")

        # Final projection to output channels
        fp = self.linear("fp", fn, IN_CHANNELS, HIDDEN)

        # Identity to output
        self.nodes.append(helper.make_node("Identity", [fp], ["out"], name="out"))

        # ═══════════════════════════════════════════════════════════════════════
        # Build graph and model
        # ═══════════════════════════════════════════════════════════════════════
        graph = helper.make_graph(
            self.nodes, "flux_dit_fp4", inputs, outputs, initializer=self.inits
        )

        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("trt", 1),
            ],
        )

        return model


def estimate_params(model: onnx.ModelProto) -> int:
    """Estimate total parameter count from initializers."""
    total = 0
    for init in model.graph.initializer:
        if init.dims:
            total += int(np.prod(init.dims))
    return total


def main():
    parser = argparse.ArgumentParser(description="Flux.1-dev DiT ONNX Generator with FP4")
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Image resolution (default: 1024)"
    )
    parser.add_argument(
        "--txt-seq", type=int, default=512, help="T5 text sequence length (default: 512)"
    )
    parser.add_argument(
        "--double",
        type=int,
        default=NUM_DOUBLE,
        help=f"Number of double-stream blocks (default: {NUM_DOUBLE})",
    )
    parser.add_argument(
        "--single",
        type=int,
        default=NUM_SINGLE,
        help=f"Number of single-stream blocks (default: {NUM_SINGLE})",
    )
    parser.add_argument("--output", type=str, default="flux_dit_fp4.onnx", help="Output file path")
    args = parser.parse_args()

    # Calculate image sequence length
    # resolution -> latent (8x downsample) -> packed (2x2) -> seq_len
    latent_size = args.resolution // 8
    packed_size = latent_size // 2
    img_seq = packed_size * packed_size

    print("=" * 70)
    print("Flux.1-dev DiT ONNX Generator with FP4 Quantization")
    print("=" * 70)
    print(f"\nGeometry:")
    print(f"  Resolution:     {args.resolution}x{args.resolution}")
    print(f"  Latent size:    {latent_size}x{latent_size}")
    print(f"  Packed (2x2):   {packed_size}x{packed_size}")
    print(f"  Image tokens:   {img_seq}")
    print(f"  Text tokens:    {args.txt_seq}")
    print(f"  Joint tokens:   {img_seq + args.txt_seq}")
    print(f"\nArchitecture:")
    print(f"  hidden_size:    {HIDDEN}")
    print(f"  num_heads:      {HEADS}")
    print(f"  head_dim:       {HEAD_DIM}")
    print(f"  mlp_hidden:     {MLP_HIDDEN}")
    print(f"  double_blocks:  {args.double}")
    print(f"  single_blocks:  {args.single}")
    print(f"\nFP4 Quantization:")
    print(f"  block_size:     {BLOCK_SIZE}")
    print(f"  weight_dtype:   FLOAT4E2M1")
    print(f"  scale_dtype:    FLOAT8E4M3FN")
    print()

    builder = FluxDiTBuilder(img_seq, args.txt_seq, args.double, args.single)
    model = builder.build()

    print(f"Saving to {args.output}...")
    onnx.save(model, args.output)

    params = estimate_params(model)
    print(f"\nModel Statistics:")
    print(f"  Nodes:          {len(model.graph.node)}")
    print(f"  Initializers:   {len(model.graph.initializer)}")
    print(f"  Parameters:     {params:,} (~{params / 1e9:.2f}B)")
    print("=" * 70)


if __name__ == "__main__":
    main()
