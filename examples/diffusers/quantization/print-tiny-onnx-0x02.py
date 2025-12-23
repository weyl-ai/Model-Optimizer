import onnx
from onnx import helper, TensorProto
import numpy as np


def make_flux_pattern_model():
    """Growing model: add 4D reshape + weight DQ path."""

    # Input BF16 - 4D like attention output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 8, 8, 64])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 8, 8, 64])

    # LayerNorm weights (BF16)
    ln_scale = helper.make_tensor(
        "ln_scale",
        TensorProto.BFLOAT16,
        [64],
        np.ones(64, dtype=np.float16).view(np.uint16).tobytes(),
        raw=True,
    )
    ln_bias = helper.make_tensor(
        "ln_bias",
        TensorProto.BFLOAT16,
        [64],
        np.zeros(64, dtype=np.float16).view(np.uint16).tobytes(),
        raw=True,
    )

    nodes = []

    # === 4D -> 3D reshape ===
    shape_3d = helper.make_tensor("shape_3d", TensorProto.INT64, [3], [1, -1, 64])

    reshape_pre = helper.make_node(
        "Reshape",
        inputs=["input", "shape_3d"],
        outputs=["input_3d"],
        name="reshape_pre_node",
    )
    nodes.append(reshape_pre)

    # === DynQuant scale constant ===
    dq_scale_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["dq_scale"],
        name="dq_scale_const_node",
        value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
    )
    nodes.append(dq_scale_const)

    # === TRT_FP4DynamicQuantize ===
    dynquant = helper.make_node(
        "TRT_FP4DynamicQuantize",
        inputs=["input_3d", "dq_scale"],
        outputs=["quantized", "scale_fp8"],
        name="dynquant_node",
        domain="trt",
        axis=-1,
        block_size=16,
        scale_type=TensorProto.FLOAT8E4M3FN,
    )
    nodes.append(dynquant)

    # === DQ1: scale FP8 -> FP32 ===
    dq1 = helper.make_node(
        "DequantizeLinear",
        inputs=["scale_fp8", "dq_scale"],
        outputs=["scale_fp32"],
        name="dq_scale_node",
        domain="trt",
    )
    nodes.append(dq1)

    # === DQ2: data FP4 -> FP32 ===
    dq2 = helper.make_node(
        "DequantizeLinear",
        inputs=["quantized", "scale_fp32"],
        outputs=["data_fp32"],
        name="dq_data_node",
        domain="trt",
        axis=-1,
        block_size=16,
    )
    nodes.append(dq2)

    # === Cast FP32 -> BF16 ===
    cast = helper.make_node(
        "Cast",
        inputs=["data_fp32"],
        outputs=["data_bf16_3d"],
        name="cast_bf16_node",
        to=TensorProto.BFLOAT16,
    )
    nodes.append(cast)

    # === 3D -> 4D reshape ===
    shape_4d = helper.make_tensor("shape_4d", TensorProto.INT64, [4], [1, 8, 8, 64])

    reshape_post = helper.make_node(
        "Reshape",
        inputs=["data_bf16_3d", "shape_4d"],
        outputs=["data_bf16"],
        name="reshape_post_node",
    )
    nodes.append(reshape_post)

    # === LayerNorm ===
    layernorm = helper.make_node(
        "LayerNormalization",
        inputs=["data_bf16", "ln_scale", "ln_bias"],
        outputs=["output"],
        name="layernorm_node",
        axis=-1,
        epsilon=1e-6,
    )
    nodes.append(layernorm)

    value_infos = [
        helper.make_tensor_value_info("dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("input_3d", TensorProto.BFLOAT16, [1, 64, 64]),
        helper.make_tensor_value_info("quantized", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("data_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("data_bf16_3d", TensorProto.BFLOAT16, [1, 64, 64]),
        helper.make_tensor_value_info("data_bf16", TensorProto.BFLOAT16, [1, 8, 8, 64]),
    ]

    graph = helper.make_graph(
        nodes,
        "flux_pattern_test",
        [input_tensor],
        [output_tensor],
        initializer=[ln_scale, ln_bias, shape_3d, shape_4d],
        value_info=value_infos,
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("trt", 1),
        ],
    )

    onnx.save(model, "flux_pattern.onnx")
    print("Saved flux_pattern.onnx")


if __name__ == "__main__":
    make_flux_pattern_model()
