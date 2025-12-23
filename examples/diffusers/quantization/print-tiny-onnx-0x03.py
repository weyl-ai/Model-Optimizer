import onnx
from onnx import helper, TensorProto
import numpy as np


def make_flux_pattern_model():
    """Test dynamic shape computation for reshape."""

    input_tensor = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 8, 8, 64])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 8, 8, 64])

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

    # Slice constants
    starts_0 = helper.make_tensor("starts_0", TensorProto.INT64, [1], [0])
    ends_1 = helper.make_tensor("ends_1", TensorProto.INT64, [1], [1])
    starts_3 = helper.make_tensor("starts_3", TensorProto.INT64, [1], [3])
    ends_4 = helper.make_tensor("ends_4", TensorProto.INT64, [1], [4])
    axes_0 = helper.make_tensor("axes_0", TensorProto.INT64, [1], [0])
    neg_one = helper.make_tensor("neg_one", TensorProto.INT64, [1], [-1])

    nodes = []

    # === Dynamic shape computation ===
    shape_node = helper.make_node(
        "Shape", inputs=["input"], outputs=["orig_shape"], name="shape_node"
    )
    nodes.append(shape_node)

    slice_dim0 = helper.make_node(
        "Slice",
        inputs=["orig_shape", "starts_0", "ends_1", "axes_0"],
        outputs=["dim0"],
        name="slice_dim0_node",
    )
    nodes.append(slice_dim0)

    slice_dim3 = helper.make_node(
        "Slice",
        inputs=["orig_shape", "starts_3", "ends_4", "axes_0"],
        outputs=["dim3"],
        name="slice_dim3_node",
    )
    nodes.append(slice_dim3)

    concat_shape = helper.make_node(
        "Concat",
        inputs=["dim0", "neg_one", "dim3"],
        outputs=["shape_3d"],
        axis=0,
        name="concat_shape_node",
    )
    nodes.append(concat_shape)

    # === Reshape 4D -> 3D (dynamic) ===
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

    # === Reshape 3D -> 4D (dynamic, use orig_shape) ===
    reshape_post = helper.make_node(
        "Reshape",
        inputs=["data_bf16_3d", "orig_shape"],
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
        helper.make_tensor_value_info("orig_shape", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("dim0", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("dim3", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("shape_3d", TensorProto.INT64, [3]),
        helper.make_tensor_value_info("input_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("quantized", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("data_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("data_bf16_3d", TensorProto.BFLOAT16, None),
        helper.make_tensor_value_info("data_bf16", TensorProto.BFLOAT16, None),
    ]

    graph = helper.make_graph(
        nodes,
        "flux_pattern_test",
        [input_tensor],
        [output_tensor],
        initializer=[ln_scale, ln_bias, starts_0, ends_1, starts_3, ends_4, axes_0, neg_one],
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
