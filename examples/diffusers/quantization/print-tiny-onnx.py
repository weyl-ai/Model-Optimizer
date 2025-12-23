import onnx
from onnx import helper, TensorProto
import numpy as np


def make_flux_pattern_model():
    """Minimal model mimicking FLUX: DynQuant -> LayerNorm."""

    # Input BF16
    input_tensor = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [1, 64, 64])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [1, 64, 64])

    # LayerNorm weights (BF16)
    ln_scale = helper.make_tensor(
        "ln_scale", TensorProto.BFLOAT16, [64], np.ones(32, dtype=np.float32).tobytes(), raw=True
    )

    ln_bias = helper.make_tensor(
        "ln_bias", TensorProto.BFLOAT16, [64], np.zeros(32, dtype=np.float32).tobytes(), raw=True
    )

    # DynQuant scale constant
    dq_scale_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["dq_scale"],
        name="dq_scale_const_node",
        value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0]),
    )

    # TRT_FP4DynamicQuantize
    dynquant = helper.make_node(
        "TRT_FP4DynamicQuantize",
        inputs=["input", "dq_scale"],
        outputs=["quantized", "scale_fp8"],
        name="dynquant_node",
        domain="trt",
        axis=-1,
        block_size=16,
        scale_type=TensorProto.FLOAT8E4M3FN,
    )

    # DQ1: scale FP8 -> FP32
    dq1 = helper.make_node(
        "DequantizeLinear",
        inputs=["scale_fp8", "dq_scale"],
        outputs=["scale_fp32"],
        name="dq_scale_node",
        domain="trt",
    )

    # DQ2: data FP4 -> FP32
    dq2 = helper.make_node(
        "DequantizeLinear",
        inputs=["quantized", "scale_fp32"],
        outputs=["data_fp32"],
        name="dq_data",
        domain="trt",
        axis=-1,
        block_size=16,
    )

    # Cast FP32 -> BF16
    cast = helper.make_node(
        "Cast",
        inputs=["data_fp32"],
        outputs=["data_bf16"],
        name="cast_bf16_node",
        to=TensorProto.BFLOAT16,
    )

    # LayerNorm (expects BF16 input, BF16 scale/bias)
    layernorm = helper.make_node(
        "LayerNormalization",
        inputs=["data_bf16", "ln_scale", "ln_bias"],
        outputs=["output"],
        name="layernorm_node",
        axis=-1,
        epsilon=1e-6,
    )

    value_infos = [
        helper.make_tensor_value_info("dq_scale", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("quantized", TensorProto.FLOAT4E2M1, None),
        helper.make_tensor_value_info("scale_fp8", TensorProto.FLOAT8E4M3FN, None),
        helper.make_tensor_value_info("scale_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("data_fp32", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("data_bf16", TensorProto.BFLOAT16, None),
    ]

    graph = helper.make_graph(
        [dq_scale_const, dynquant, dq1, dq2, cast, layernorm],
        "flux_pattern_test",
        [input_tensor],
        [output_tensor],
        initializer=[ln_scale, ln_bias],
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
