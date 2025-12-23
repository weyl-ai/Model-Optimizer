import onnx
from onnx import helper, TensorProto
import numpy as np


def make_fp4_attention_model():
    """Full FP4 attention: dynquant on Q/K/V activations + FP4 weights."""

    input_tensor = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 64, 64])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 64, 64])

    # QKV weights FP4
    qkv_w_data = np.random.randint(0, 255, (192, 32), dtype=np.uint8).tobytes()
    qkv_w = helper.make_tensor("qkv_w", TensorProto.FLOAT4E2M1, [192, 64], qkv_w_data, raw=True)
    qkv_scale_data = np.ones((192, 4), dtype=np.uint8).tobytes()
    qkv_scale = helper.make_tensor(
        "qkv_scale", TensorProto.FLOAT8E4M3FN, [192, 4], qkv_scale_data, raw=True
    )
    qkv_global = helper.make_tensor("qkv_global", TensorProto.FLOAT, [], [1.0])

    # Output proj FP4
    out_w_data = np.random.randint(0, 255, (64, 32), dtype=np.uint8).tobytes()
    out_w = helper.make_tensor("out_w", TensorProto.FLOAT4E2M1, [64, 64], out_w_data, raw=True)
    out_scale_data = np.ones((64, 4), dtype=np.uint8).tobytes()
    out_scale = helper.make_tensor(
        "out_scale", TensorProto.FLOAT8E4M3FN, [64, 4], out_scale_data, raw=True
    )
    out_global = helper.make_tensor("out_global", TensorProto.FLOAT, [], [1.0])

    # Shape tensors
    shape_4d = helper.make_tensor("shape_4d", TensorProto.INT64, [4], [1, 64, 8, 8])
    shape_3d = helper.make_tensor("shape_3d", TensorProto.INT64, [3], [1, 64, 64])
    shape_bmm = helper.make_tensor(
        "shape_bmm", TensorProto.INT64, [3], [8, 64, 8]
    )  # [heads, seq, head_dim]
    shape_bmm_scores = helper.make_tensor("shape_bmm_scores", TensorProto.INT64, [3], [8, 64, 64])
    shape_4d_scores = helper.make_tensor("shape_4d_scores", TensorProto.INT64, [4], [1, 8, 64, 64])
    shape_4d_out = helper.make_tensor("shape_4d_out", TensorProto.INT64, [4], [1, 8, 64, 8])

    scale_val = helper.make_tensor("scale_val", TensorProto.BFLOAT16, [], [0.35355339])

    nodes = []

    # === QKV weight DQ ===
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["qkv_scale", "qkv_global"],
            ["qkv_scale_fp32"],
            name="qkv_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["qkv_w", "qkv_scale_fp32"],
            ["qkv_w_fp32"],
            name="qkv_dq_w_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", ["qkv_w_fp32"], ["qkv_w_bf16"], name="qkv_cast_node", to=TensorProto.BFLOAT16
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose", ["qkv_w_bf16"], ["qkv_w_t"], name="qkv_transpose_node", perm=[1, 0]
        )
    )

    # === Input activation dynquant ===
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["input_dq_scale"],
            name="input_dq_scale_const",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["input", "input_dq_scale"],
            ["input_q", "input_scale_fp8"],
            name="input_dynquant_node",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=TensorProto.FLOAT8E4M3FN,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["input_scale_fp8", "input_dq_scale"],
            ["input_scale_fp32"],
            name="input_dq_scale_act_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale_fp32"],
            ["input_dq_fp32"],
            name="input_dq_data_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast",
            ["input_dq_fp32"],
            ["input_dq_bf16"],
            name="input_cast_node",
            to=TensorProto.BFLOAT16,
        )
    )

    # === QKV projection ===
    nodes.append(
        helper.make_node(
            "MatMul", ["input_dq_bf16", "qkv_w_t"], ["qkv_out"], name="qkv_matmul_node"
        )
    )

    # Split Q, K, V
    nodes.append(
        helper.make_node(
            "Split", ["qkv_out"], ["q", "k", "v"], name="qkv_split_node", axis=-1, num_outputs=3
        )
    )

    # === Reshape & transpose for attention ===
    nodes.append(helper.make_node("Reshape", ["q", "shape_4d"], ["q_4d"], name="q_reshape_node"))
    nodes.append(helper.make_node("Reshape", ["k", "shape_4d"], ["k_4d"], name="k_reshape_node"))
    nodes.append(helper.make_node("Reshape", ["v", "shape_4d"], ["v_4d"], name="v_reshape_node"))

    nodes.append(
        helper.make_node("Transpose", ["q_4d"], ["q_t"], name="q_transpose_node", perm=[0, 2, 1, 3])
    )
    nodes.append(
        helper.make_node("Transpose", ["k_4d"], ["k_t"], name="k_transpose_node", perm=[0, 2, 1, 3])
    )
    nodes.append(
        helper.make_node("Transpose", ["v_4d"], ["v_t"], name="v_transpose_node", perm=[0, 2, 1, 3])
    )

    # === Q activation dynquant (reshape 4D->3D for TRT) ===
    nodes.append(helper.make_node("Reshape", ["q_t", "shape_bmm"], ["q_3d"], name="q_to_3d_node"))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["q_dq_scale"],
            name="q_dq_scale_const",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["q_3d", "q_dq_scale"],
            ["q_fp4", "q_scale_fp8"],
            name="q_dynquant_node",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=TensorProto.FLOAT8E4M3FN,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["q_scale_fp8", "q_dq_scale"],
            ["q_scale_fp32"],
            name="q_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["q_fp4", "q_scale_fp32"],
            ["q_dq_fp32"],
            name="q_dq_data_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", ["q_dq_fp32"], ["q_dq_bf16"], name="q_cast_node", to=TensorProto.BFLOAT16
        )
    )
    nodes.append(
        helper.make_node("Reshape", ["q_dq_bf16", "shape_4d_out"], ["q_ready"], name="q_to_4d_node")
    )

    # === K activation dynquant ===
    nodes.append(helper.make_node("Reshape", ["k_t", "shape_bmm"], ["k_3d"], name="k_to_3d_node"))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["k_dq_scale"],
            name="k_dq_scale_const",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["k_3d", "k_dq_scale"],
            ["k_fp4", "k_scale_fp8"],
            name="k_dynquant_node",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=TensorProto.FLOAT8E4M3FN,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["k_scale_fp8", "k_dq_scale"],
            ["k_scale_fp32"],
            name="k_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["k_fp4", "k_scale_fp32"],
            ["k_dq_fp32"],
            name="k_dq_data_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", ["k_dq_fp32"], ["k_dq_bf16"], name="k_cast_node", to=TensorProto.BFLOAT16
        )
    )
    nodes.append(
        helper.make_node("Reshape", ["k_dq_bf16", "shape_4d_out"], ["k_ready"], name="k_to_4d_node")
    )

    # K transpose for BMM1
    nodes.append(
        helper.make_node(
            "Transpose", ["k_ready"], ["k_ready_t"], name="k_transpose_bmm_node", perm=[0, 1, 3, 2]
        )
    )

    # === V activation dynquant ===
    nodes.append(helper.make_node("Reshape", ["v_t", "shape_bmm"], ["v_3d"], name="v_to_3d_node"))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["v_dq_scale"],
            name="v_dq_scale_const",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["v_3d", "v_dq_scale"],
            ["v_fp4", "v_scale_fp8"],
            name="v_dynquant_node",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=TensorProto.FLOAT8E4M3FN,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["v_scale_fp8", "v_dq_scale"],
            ["v_scale_fp32"],
            name="v_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["v_fp4", "v_scale_fp32"],
            ["v_dq_fp32"],
            name="v_dq_data_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", ["v_dq_fp32"], ["v_dq_bf16"], name="v_cast_node", to=TensorProto.BFLOAT16
        )
    )
    nodes.append(
        helper.make_node("Reshape", ["v_dq_bf16", "shape_4d_out"], ["v_ready"], name="v_to_4d_node")
    )

    # === BMM1: Q @ K^T ===
    nodes.append(
        helper.make_node("MatMul", ["q_ready", "k_ready_t"], ["attn_scores"], name="bmm1_node")
    )

    # Scale + Softmax
    nodes.append(
        helper.make_node(
            "Mul", ["attn_scores", "scale_val"], ["attn_scaled"], name="scale_attn_node"
        )
    )
    nodes.append(
        helper.make_node("Softmax", ["attn_scaled"], ["attn_weights"], name="softmax_node", axis=-1)
    )

    # === Softmax output dynquant ===
    nodes.append(
        helper.make_node(
            "Reshape", ["attn_weights", "shape_bmm_scores"], ["attn_3d"], name="attn_to_3d_node"
        )
    )
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["attn_dq_scale"],
            name="attn_dq_scale_const",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["attn_3d", "attn_dq_scale"],
            ["attn_fp4", "attn_scale_fp8"],
            name="attn_dynquant_node",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=TensorProto.FLOAT8E4M3FN,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["attn_scale_fp8", "attn_dq_scale"],
            ["attn_scale_fp32"],
            name="attn_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["attn_fp4", "attn_scale_fp32"],
            ["attn_dq_fp32"],
            name="attn_dq_data_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast",
            ["attn_dq_fp32"],
            ["attn_dq_bf16"],
            name="attn_cast_node",
            to=TensorProto.BFLOAT16,
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ["attn_dq_bf16", "shape_4d_scores"], ["attn_ready"], name="attn_to_4d_node"
        )
    )

    # === BMM2: attn @ V ===
    nodes.append(
        helper.make_node("MatMul", ["attn_ready", "v_ready"], ["bmm2_out"], name="bmm2_node")
    )

    # === BMM2 output dynquant ===
    nodes.append(
        helper.make_node("Reshape", ["bmm2_out", "shape_bmm"], ["bmm2_3d"], name="bmm2_to_3d_node")
    )
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["bmm2_dq_scale"],
            name="bmm2_dq_scale_const",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["bmm2_3d", "bmm2_dq_scale"],
            ["bmm2_fp4", "bmm2_scale_fp8"],
            name="bmm2_dynquant_node",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=TensorProto.FLOAT8E4M3FN,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["bmm2_scale_fp8", "bmm2_dq_scale"],
            ["bmm2_scale_fp32"],
            name="bmm2_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["bmm2_fp4", "bmm2_scale_fp32"],
            ["bmm2_dq_fp32"],
            name="bmm2_dq_data_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast",
            ["bmm2_dq_fp32"],
            ["bmm2_dq_bf16"],
            name="bmm2_cast_node",
            to=TensorProto.BFLOAT16,
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ["bmm2_dq_bf16", "shape_4d_out"], ["bmm2_4d"], name="bmm2_to_4d_node"
        )
    )

    # Transpose back & flatten
    nodes.append(
        helper.make_node(
            "Transpose",
            ["bmm2_4d"],
            ["attn_out_t"],
            name="attn_transpose_back_node",
            perm=[0, 2, 1, 3],
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ["attn_out_t", "shape_3d"], ["attn_flat"], name="attn_reshape_node"
        )
    )

    # === Output proj weight DQ ===
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["out_scale", "out_global"],
            ["out_scale_fp32"],
            name="out_dq_scale_node",
            domain="trt",
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            ["out_w", "out_scale_fp32"],
            ["out_w_fp32"],
            name="out_dq_w_node",
            domain="trt",
            axis=-1,
            block_size=16,
        )
    )
    nodes.append(
        helper.make_node(
            "Cast", ["out_w_fp32"], ["out_w_bf16"], name="out_cast_node", to=TensorProto.BFLOAT16
        )
    )
    nodes.append(
        helper.make_node(
            "Transpose", ["out_w_bf16"], ["out_w_t"], name="out_transpose_node", perm=[1, 0]
        )
    )

    # === Output projection ===
    nodes.append(
        helper.make_node("MatMul", ["attn_flat", "out_w_t"], ["output"], name="out_matmul_node")
    )

    value_infos = [
        helper.make_tensor_value_info("qkv_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("qkv_w_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("qkv_w_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("qkv_w_t", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("input_dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("input_q", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("input_scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("input_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("input_dq_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("input_dq_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("qkv_out", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("q", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("v", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("q_4d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k_4d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("v_4d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("q_t", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k_t", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("v_t", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("q_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("q_dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("q_fp4", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("q_scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("q_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("q_dq_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("q_dq_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("q_ready", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k_dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("k_fp4", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("k_scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("k_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("k_dq_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("k_dq_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k_ready", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("k_ready_t", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("v_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("v_dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("v_fp4", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("v_scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("v_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("v_dq_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("v_dq_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("v_ready", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_scores", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_scaled", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_weights", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("attn_fp4", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("attn_scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("attn_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("attn_dq_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("attn_dq_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_ready", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("bmm2_out", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("bmm2_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("bmm2_dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("bmm2_fp4", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("bmm2_scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("bmm2_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("bmm2_dq_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("bmm2_dq_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("bmm2_4d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_out_t", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("attn_flat", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("out_scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("out_w_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("out_w_bf16", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("out_w_t", TensorProto.BFLOAT16, None),
    ]

    graph = helper.make_graph(
        nodes,
        "fp4_attention_test",
        [input_tensor],
        [output_tensor],
        initializer=[
            qkv_w,
            qkv_scale,
            qkv_global,
            out_w,
            out_scale,
            out_global,
            shape_4d,
            shape_3d,
            shape_bmm,
            shape_bmm_scores,
            shape_4d_scores,
            shape_4d_out,
            scale_val,
        ],
        value_info=value_infos,
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("trt", 1),
        ],
    )

    onnx.save(model, "fp4_attention.onnx")
    print("Saved fp4_attention.onnx")


if __name__ == "__main__":
    make_fp4_attention_model()
