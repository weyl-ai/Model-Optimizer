import onnx
from onnx import helper, TensorProto
import numpy as np


def make_transformer_block():
    """
    Full transformer block: Attention + MLP + Residuals + LayerNorms, all FP4.

    Resized to Flux.1-dev-like per-block geometry:
      - heads=24
      - head_dim=128
      - hidden=heads*head_dim=3072
      - seq_len=4096 (typical latent tokens)
    Sources: FluxTransformer2DModel defaults for heads/head_dim, and typical hidden_states token count. :contentReference[oaicite:1]{index=1}
    """

    # === Flux-like sizes ===
    seq_len = 4096
    heads = 24
    head_dim = 128
    hidden = heads * head_dim  # 3072
    mlp_hidden = hidden * 4  # 12288

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.BFLOAT16, [1, seq_len, hidden]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.BFLOAT16, [1, seq_len, hidden]
    )

    def bf16_raw_from_f32(x: float) -> bytes:
        """Encode float32 scalar into BF16 raw bytes."""
        u16 = (np.float32(x).view(np.uint32) >> 16).astype(np.uint16)
        return u16.tobytes()

    def make_fp4_weight(name, out_dim, in_dim):
        """Create FP4 weight + scale tensors."""
        # Store FP4E2M1 as packed bytes: (out_dim, in_dim//2) bytes
        w_data = np.random.randint(0, 255, (out_dim, in_dim // 2), dtype=np.uint8).tobytes()
        w = helper.make_tensor(
            f"{name}_w", TensorProto.FLOAT4E2M1, [out_dim, in_dim], w_data, raw=True
        )

        # Per-block scale (block_size=16): shape (out_dim, in_dim//16) of float8 bytes
        s_data = np.ones((out_dim, in_dim // 16), dtype=np.uint8).tobytes()
        s = helper.make_tensor(
            f"{name}_s", TensorProto.FLOAT8E4M3FN, [out_dim, in_dim // 16], s_data, raw=True
        )

        # Global scale factor (float)
        g = helper.make_tensor(f"{name}_g", TensorProto.FLOAT, [], [1.0])
        return w, s, g

    def make_ln_params(name, dim):
        """Create LayerNorm scale/bias as BF16."""
        scale_f32 = np.ones(dim, dtype=np.float32)
        scale_bf16 = (scale_f32.view(np.uint32) >> 16).astype(np.uint16)
        scale = helper.make_tensor(
            f"{name}_scale", TensorProto.BFLOAT16, [dim], scale_bf16.tobytes(), raw=True
        )

        bias_f32 = np.zeros(dim, dtype=np.float32)
        bias_bf16 = (bias_f32.view(np.uint32) >> 16).astype(np.uint16)
        bias = helper.make_tensor(
            f"{name}_bias", TensorProto.BFLOAT16, [dim], bias_bf16.tobytes(), raw=True
        )
        return scale, bias

    # Weights
    qkv_w, qkv_s, qkv_g = make_fp4_weight("qkv", hidden * 3, hidden)  # (9216, 3072)
    out_w, out_s, out_g = make_fp4_weight("out", hidden, hidden)  # (3072, 3072)
    mlp_up_w, mlp_up_s, mlp_up_g = make_fp4_weight("mlp_up", mlp_hidden, hidden)  # (12288, 3072)
    mlp_down_w, mlp_down_s, mlp_down_g = make_fp4_weight(
        "mlp_down", hidden, mlp_hidden
    )  # (3072, 12288)

    ln1_scale, ln1_bias = make_ln_params("ln1", hidden)
    ln2_scale, ln2_bias = make_ln_params("ln2", hidden)

    # Shape tensors (all seq_len-aware)
    shape_attn = helper.make_tensor(
        "shape_attn", TensorProto.INT64, [4], [1, seq_len, heads, head_dim]
    )
    shape_flat = helper.make_tensor("shape_flat", TensorProto.INT64, [3], [1, seq_len, hidden])
    shape_bmm = helper.make_tensor("shape_bmm", TensorProto.INT64, [3], [heads, seq_len, head_dim])
    shape_scores = helper.make_tensor(
        "shape_scores", TensorProto.INT64, [3], [heads, seq_len, seq_len]
    )
    shape_4d_scores = helper.make_tensor(
        "shape_4d_scores", TensorProto.INT64, [4], [1, heads, seq_len, seq_len]
    )
    shape_4d_out = helper.make_tensor(
        "shape_4d_out", TensorProto.INT64, [4], [1, heads, seq_len, head_dim]
    )

    # Attention scale (BF16 scalar)
    scale_val = helper.make_tensor(
        "scale_val", TensorProto.BFLOAT16, [], bf16_raw_from_f32(1.0 / np.sqrt(head_dim)), raw=True
    )

    nodes = []

    def add_weight_dq(prefix, w_name, s_name, g_name, out_name):
        """Add weight dequantization pattern."""
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [s_name, g_name],
                [f"{prefix}_s_fp32"],
                name=f"{prefix}_dq_s",
                domain="trt",
            )
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [w_name, f"{prefix}_s_fp32"],
                [f"{prefix}_w_fp32"],
                name=f"{prefix}_dq_w",
                domain="trt",
                axis=-1,
                block_size=16,
            )
        )
        nodes.append(
            helper.make_node(
                "Cast",
                [f"{prefix}_w_fp32"],
                [f"{prefix}_w_bf16"],
                name=f"{prefix}_cast",
                to=TensorProto.BFLOAT16,
            )
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{prefix}_w_bf16"],
                [out_name],
                name=f"{prefix}_transpose",
                perm=[1, 0],
            )
        )

    def add_act_dynquant(prefix, input_name, output_name):
        """Add activation dynamic quantization pattern."""
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [f"{prefix}_dq_scale"],
                name=f"{prefix}_dq_scale_const",
                value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
            )
        )
        nodes.append(
            helper.make_node(
                "TRT_FP4DynamicQuantize",
                [input_name, f"{prefix}_dq_scale"],
                [f"{prefix}_fp4", f"{prefix}_scale_fp8"],
                name=f"{prefix}_dynquant",
                domain="trt",
                axis=-1,
                block_size=16,
                scale_type=TensorProto.FLOAT8E4M3FN,
            )
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{prefix}_scale_fp8", f"{prefix}_dq_scale"],
                [f"{prefix}_scale_fp32"],
                name=f"{prefix}_dq_scale_act",
                domain="trt",
            )
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [f"{prefix}_fp4", f"{prefix}_scale_fp32"],
                [f"{prefix}_dq_fp32"],
                name=f"{prefix}_dq_data",
                domain="trt",
                axis=-1,
                block_size=16,
            )
        )
        nodes.append(
            helper.make_node(
                "Cast",
                [f"{prefix}_dq_fp32"],
                [output_name],
                name=f"{prefix}_cast_bf16",
                to=TensorProto.BFLOAT16,
            )
        )

    # === Pre-attention LayerNorm ===
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["input", "ln1_scale", "ln1_bias"],
            ["ln1_out"],
            name="ln1",
            axis=-1,
            epsilon=1e-5,
        )
    )

    # === QKV Projection ===
    add_weight_dq("qkv", "qkv_w", "qkv_s", "qkv_g", "qkv_w_t")
    add_act_dynquant("qkv_act", "ln1_out", "qkv_act_bf16")
    nodes.append(
        helper.make_node("MatMul", ["qkv_act_bf16", "qkv_w_t"], ["qkv_out"], name="qkv_matmul")
    )

    # Split Q, K, V
    nodes.append(
        helper.make_node(
            "Split", ["qkv_out"], ["q", "k", "v"], name="qkv_split", axis=-1, num_outputs=3
        )
    )

    # Reshape & transpose for attention
    for name in ["q", "k", "v"]:
        nodes.append(
            helper.make_node(
                "Reshape", [name, "shape_attn"], [f"{name}_4d"], name=f"{name}_reshape"
            )
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{name}_4d"],
                [f"{name}_t"],
                name=f"{name}_transpose",
                perm=[0, 2, 1, 3],
            )
        )

    # === Q activation quantize for BMM ===
    nodes.append(helper.make_node("Reshape", ["q_t", "shape_bmm"], ["q_3d"], name="q_to_3d"))
    add_act_dynquant("q", "q_3d", "q_dq_bf16")
    nodes.append(
        helper.make_node("Reshape", ["q_dq_bf16", "shape_4d_out"], ["q_ready"], name="q_to_4d")
    )

    # === K activation quantize ===
    nodes.append(helper.make_node("Reshape", ["k_t", "shape_bmm"], ["k_3d"], name="k_to_3d"))
    add_act_dynquant("k", "k_3d", "k_dq_bf16")
    nodes.append(
        helper.make_node("Reshape", ["k_dq_bf16", "shape_4d_out"], ["k_ready"], name="k_to_4d")
    )
    nodes.append(
        helper.make_node(
            "Transpose", ["k_ready"], ["k_ready_t"], name="k_transpose_bmm", perm=[0, 1, 3, 2]
        )
    )

    # === V activation quantize ===
    nodes.append(helper.make_node("Reshape", ["v_t", "shape_bmm"], ["v_3d"], name="v_to_3d"))
    add_act_dynquant("v", "v_3d", "v_dq_bf16")
    nodes.append(
        helper.make_node("Reshape", ["v_dq_bf16", "shape_4d_out"], ["v_ready"], name="v_to_4d")
    )

    # === BMM1: Q @ K^T ===
    nodes.append(helper.make_node("MatMul", ["q_ready", "k_ready_t"], ["scores"], name="bmm1"))
    nodes.append(helper.make_node("Mul", ["scores", "scale_val"], ["scores_scaled"], name="scale"))
    nodes.append(
        helper.make_node("Softmax", ["scores_scaled"], ["attn_weights"], name="softmax", axis=-1)
    )

    # === Softmax output quantize ===
    nodes.append(
        helper.make_node(
            "Reshape", ["attn_weights", "shape_scores"], ["attn_3d"], name="attn_to_3d"
        )
    )
    add_act_dynquant("attn", "attn_3d", "attn_dq_bf16")
    nodes.append(
        helper.make_node(
            "Reshape", ["attn_dq_bf16", "shape_4d_scores"], ["attn_ready"], name="attn_to_4d"
        )
    )

    # === BMM2: attn @ V ===
    nodes.append(helper.make_node("MatMul", ["attn_ready", "v_ready"], ["bmm2_out"], name="bmm2"))

    # === BMM2 output quantize ===
    nodes.append(
        helper.make_node("Reshape", ["bmm2_out", "shape_bmm"], ["bmm2_3d"], name="bmm2_to_3d")
    )
    add_act_dynquant("bmm2", "bmm2_3d", "bmm2_dq_bf16")
    nodes.append(
        helper.make_node(
            "Reshape", ["bmm2_dq_bf16", "shape_4d_out"], ["bmm2_4d"], name="bmm2_to_4d"
        )
    )

    # Transpose back & flatten
    nodes.append(
        helper.make_node(
            "Transpose", ["bmm2_4d"], ["attn_out_t"], name="attn_transpose_back", perm=[0, 2, 1, 3]
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ["attn_out_t", "shape_flat"], ["attn_flat"], name="attn_flatten"
        )
    )

    # === Output projection ===
    add_weight_dq("out", "out_w", "out_s", "out_g", "out_w_t")
    add_act_dynquant("out_act", "attn_flat", "out_act_bf16")
    nodes.append(
        helper.make_node("MatMul", ["out_act_bf16", "out_w_t"], ["attn_proj"], name="out_matmul")
    )

    # === Residual 1 ===
    nodes.append(helper.make_node("Add", ["input", "attn_proj"], ["res1"], name="residual1"))

    # === Pre-MLP LayerNorm ===
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["res1", "ln2_scale", "ln2_bias"],
            ["ln2_out"],
            name="ln2",
            axis=-1,
            epsilon=1e-5,
        )
    )

    # === MLP Up ===
    add_weight_dq("mlp_up", "mlp_up_w", "mlp_up_s", "mlp_up_g", "mlp_up_w_t")
    add_act_dynquant("mlp_up_act", "ln2_out", "mlp_up_act_bf16")
    nodes.append(
        helper.make_node(
            "MatMul", ["mlp_up_act_bf16", "mlp_up_w_t"], ["mlp_up_out"], name="mlp_up_matmul"
        )
    )

    # GELU
    nodes.append(helper.make_node("Gelu", ["mlp_up_out"], ["mlp_gelu"], name="gelu", domain=""))

    # === MLP Down ===
    add_weight_dq("mlp_down", "mlp_down_w", "mlp_down_s", "mlp_down_g", "mlp_down_w_t")
    add_act_dynquant("mlp_down_act", "mlp_gelu", "mlp_down_act_bf16")
    nodes.append(
        helper.make_node(
            "MatMul", ["mlp_down_act_bf16", "mlp_down_w_t"], ["mlp_out"], name="mlp_down_matmul"
        )
    )

    # === Residual 2 ===
    nodes.append(helper.make_node("Add", ["res1", "mlp_out"], ["output"], name="residual2"))

    initializers = [
        qkv_w,
        qkv_s,
        qkv_g,
        out_w,
        out_s,
        out_g,
        mlp_up_w,
        mlp_up_s,
        mlp_up_g,
        mlp_down_w,
        mlp_down_s,
        mlp_down_g,
        ln1_scale,
        ln1_bias,
        ln2_scale,
        ln2_bias,
        shape_attn,
        shape_flat,
        shape_bmm,
        shape_scores,
        shape_4d_scores,
        shape_4d_out,
        scale_val,
    ]

    graph = helper.make_graph(
        nodes, "transformer_block", [input_tensor], [output_tensor], initializer=initializers
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("trt", 1)],
    )

    onnx.save(model, "transformer_block.onnx")
    print("Saved transformer_block.onnx")


if __name__ == "__main__":
    make_transformer_block()
