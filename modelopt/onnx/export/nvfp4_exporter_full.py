# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVFP4 quantization exporter (fixed)."""

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from modelopt.onnx import utils
from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_nodes
from modelopt.onnx.quantization.qdq_utils import onnx_dtype_map
from modelopt.onnx.quantization.quant_utils import (
    get_weights_scaling_factor,
    get_weights_scaling_factor_2,
    quantize,
)
from modelopt.torch.quantization.qtensor import NVFP4QTensor

from .base_exporter import ONNXQuantExporter


def _cast_fp4(array: np.ndarray) -> np.ndarray:
    """Cast a numpy array to FLOAT4E2M1 using PyTorch."""
    array_f32_t = torch.from_numpy(array)
    array_f32_t_shape = array_f32_t.shape
    assert array_f32_t_shape[-1] % 2 == 0, "last dimension must be divisible by 2"

    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()

    array_f4_t = NVFP4QTensor._cast_fp4(array_f32_t)
    array_f4_t_packed = array_f4_t[..., ::2] | (array_f4_t[..., 1::2] << 4)
    array_f4 = array_f4_t_packed.cpu().numpy().astype(np.uint8)
    return array_f4


def _cast_fp8(array: np.ndarray) -> np.ndarray:
    """Cast a numpy array to FLOAT8E4M3FN using PyTorch."""
    array_f32_t = torch.from_numpy(array)
    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()
    array_f8_t = array_f32_t.clamp(min=-448, max=448).to(torch.float8_e4m3fn).view(torch.uint8)
    array_f8 = array_f8_t.cpu().numpy().astype(np.uint8)
    return array_f8


def _get_initializer_value(graph: onnx.GraphProto, name: str) -> np.ndarray | None:
    """Get numpy array from initializer by name."""
    for init in graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None


def _get_value_info_shape(
    value_info_map: dict[str, onnx.ValueInfoProto], name: str
) -> list[int] | None:
    """Extract shape from value_info, returns None if unknown."""
    if name not in value_info_map:
        return None
    vi = value_info_map[name]
    if not vi.type.HasField("tensor_type"):
        return None
    shape = vi.type.tensor_type.shape
    if not shape.dim:
        return None
    dims = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            dims.append(-1)
    return dims


def _collect_existing_names(graph: onnx.GraphProto) -> set[str]:
    """Collect all existing names in the graph."""
    existing_names = set()
    for node in graph.node:
        if node.name:
            existing_names.add(node.name)
        existing_names.update(node.input)
        existing_names.update(node.output)
    for init in graph.initializer:
        existing_names.add(init.name)
    for vi in graph.value_info:
        existing_names.add(vi.name)
    for inp in graph.input:
        existing_names.add(inp.name)
    for out in graph.output:
        existing_names.add(out.name)
    return existing_names


def _reshape_for_dynquant(
    graph: onnx.GraphProto,
    input_name: str,
    output_name: str,
    node_name: str,
    existing_names: set[str],
) -> tuple[str, str, list[onnx.NodeProto], list[onnx.TensorProto]]:
    """Wrap 4D tensor with reshape to 3D and back for TRT_FP4DynamicQuantize."""

    original_shape_name = f"{node_name}_original_shape"
    input_3d_name = f"{node_name}_input_3d"
    output_3d_name = f"{node_name}_output_3d"
    shape_3d_name = f"{node_name}_shape_3d"
    dim0_name = f"{node_name}_dim0"
    dim3_name = f"{node_name}_dim3"

    starts_0_name = f"{node_name}_starts_0"
    ends_1_name = f"{node_name}_ends_1"
    starts_3_name = f"{node_name}_starts_3"
    ends_4_name = f"{node_name}_ends_4"
    axes_0_name = f"{node_name}_axes_0"
    neg_one_name = f"{node_name}_neg_one"

    initializers = [
        onnx.helper.make_tensor(starts_0_name, onnx.TensorProto.INT64, [1], [0]),
        onnx.helper.make_tensor(ends_1_name, onnx.TensorProto.INT64, [1], [1]),
        onnx.helper.make_tensor(starts_3_name, onnx.TensorProto.INT64, [1], [3]),
        onnx.helper.make_tensor(ends_4_name, onnx.TensorProto.INT64, [1], [4]),
        onnx.helper.make_tensor(axes_0_name, onnx.TensorProto.INT64, [1], [0]),
        onnx.helper.make_tensor(neg_one_name, onnx.TensorProto.INT64, [1], [-1]),
    ]

    shape_node = onnx.helper.make_node(
        "Shape", inputs=[input_name], outputs=[original_shape_name], name=f"{node_name}_Shape"
    )
    slice_dim0 = onnx.helper.make_node(
        "Slice",
        inputs=[original_shape_name, starts_0_name, ends_1_name, axes_0_name],
        outputs=[dim0_name],
        name=f"{node_name}_Slice_dim0",
    )
    slice_dim3 = onnx.helper.make_node(
        "Slice",
        inputs=[original_shape_name, starts_3_name, ends_4_name, axes_0_name],
        outputs=[dim3_name],
        name=f"{node_name}_Slice_dim3",
    )
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=[dim0_name, neg_one_name, dim3_name],
        outputs=[shape_3d_name],
        axis=0,
        name=f"{node_name}_Concat_shape",
    )
    reshape_pre = onnx.helper.make_node(
        "Reshape",
        inputs=[input_name, shape_3d_name],
        outputs=[input_3d_name],
        name=f"{node_name}_Reshape_pre",
    )
    reshape_post = onnx.helper.make_node(
        "Reshape",
        inputs=[output_3d_name, original_shape_name],
        outputs=[output_name],
        name=f"{node_name}_Reshape_post",
    )

    nodes = [shape_node, slice_dim0, slice_dim3, concat_node, reshape_pre, reshape_post]

    existing_names.update(
        [
            original_shape_name,
            input_3d_name,
            output_3d_name,
            shape_3d_name,
            dim0_name,
            dim3_name,
            starts_0_name,
            ends_1_name,
            starts_3_name,
            ends_4_name,
            axes_0_name,
            neg_one_name,
        ]
        + [n.name for n in nodes]
    )

    return input_3d_name, output_3d_name, nodes, initializers


def _replace_weight_fp4qdq_with_2dq(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    node_idx: int,
    initializer_indices: dict[str, int],
    value_info_map: dict[str, onnx.ValueInfoProto],
    graph_inputs: set[str],
    existing_names: set[str],
    w_f4: np.ndarray,
    sw_f32_per_tensor: np.ndarray,
    sw_f8_per_block: np.ndarray,
    block_size: int,
):
    """Replaces a weight FP4QDQ node with two trt::DequantizeLinear nodes."""

    def _add_initializer(initializer):
        if initializer.name not in initializer_indices:
            graph.initializer.append(initializer)
            existing_names.add(initializer.name)

    def _add_input_value_info(graph, tensor_proto):
        if tensor_proto.name in graph_inputs or tensor_proto.name in value_info_map:
            return
        value_info = onnx.helper.make_tensor_value_info(
            tensor_proto.name, tensor_proto.data_type, tensor_proto.dims
        )
        graph.input.append(value_info)
        graph_inputs.add(tensor_proto.name)

    graph.node.remove(node)

    w_f4_name = f"{node.name}_w_f4"
    sw_f8_per_block_name = f"{node.name}_sw_f8_scale"
    sw_f32_per_tensor_name = f"{node.name}_sw_f32_per_tensor"
    sw_f32_name = f"{node.name}_sw_f32_per_block"

    unpacked_shape = [*w_f4.shape[:-1], w_f4.shape[-1] * 2]

    w_f4_proto = onnx.helper.make_tensor(
        name=w_f4_name,
        data_type=onnx.TensorProto.FLOAT4E2M1,
        dims=unpacked_shape,
        vals=w_f4.tobytes(),
        raw=True,
    )
    sw_f32_per_tensor_proto = onnx.numpy_helper.from_array(
        sw_f32_per_tensor, sw_f32_per_tensor_name
    )
    sw_f8_per_block_proto = onnx.helper.make_tensor(
        name=sw_f8_per_block_name,
        data_type=onnx.TensorProto.FLOAT8E4M3FN,
        dims=[*sw_f8_per_block.shape],
        vals=sw_f8_per_block.tobytes(),
        raw=True,
    )

    _add_input_value_info(graph, w_f4_proto)
    _add_input_value_info(graph, sw_f32_per_tensor_proto)
    _add_input_value_info(graph, sw_f8_per_block_proto)

    _add_initializer(w_f4_proto)
    _add_initializer(sw_f32_per_tensor_proto)
    _add_initializer(sw_f8_per_block_proto)

    # DQ1: sw_f8 * sw_f32_per_tensor -> sw_f32 (trt domain, no axis/block_size)
    dequant1 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[sw_f8_per_block_proto.name, sw_f32_per_tensor_proto.name],
        outputs=[sw_f32_name],
        name=f"{node.name}_DequantizeLinear_sw",
        domain="trt",
    )

    # DQ2: w_f4 * sw_f32 -> w_f32 (trt domain, with axis/block_size)
    w32_name = node.output[0]
    dequant2 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[w_f4_proto.name, sw_f32_name],
        outputs=[w32_name],
        name=f"{node.name}_DequantizeLinear_w",
        domain="trt",
        axis=-1,
        block_size=block_size,
    )

    # Value info for intermediate scale (FP32)
    sw_f32_value_info = onnx.helper.make_tensor_value_info(
        sw_f32_name, onnx.TensorProto.FLOAT, list(sw_f8_per_block.shape)
    )

    graph.value_info.append(sw_f32_value_info)
    value_info_map[sw_f32_name] = sw_f32_value_info

    # Output is FP32
    if w32_name in value_info_map:
        value_info_map[w32_name].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    else:
        vi = onnx.helper.make_tensor_value_info(w32_name, onnx.TensorProto.FLOAT, None)
        graph.value_info.append(vi)
        value_info_map[w32_name] = vi

    graph.node.extend([dequant1, dequant2])
    existing_names.update([dequant1.name, dequant2.name, sw_f32_name])


def _replace_activation_fp4qdq_with_dynquant(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    node_idx: int,
    value_info_map: dict[str, onnx.ValueInfoProto],
    existing_names: set[str],
    block_size: int,
):
    """Replaces an activation FP4QDQ node with TRT_FP4DynamicQuantize + trt::DequantizeLinear."""
    input_name = node.input[0]
    output_name = node.output[0]
    input_shape = _get_value_info_shape(value_info_map, input_name)
    needs_reshape = input_shape is not None and len(input_shape) == 4

    # Get original output type
    original_output_type = onnx.TensorProto.BFLOAT16
    if output_name in value_info_map:
        original_output_type = value_info_map[output_name].type.tensor_type.elem_type

    # Extract amax if present
    amax = None
    if len(node.input) > 1:
        amax_tensor = _get_initializer_value(graph, node.input[1])
        if amax_tensor is not None:
            amax = float(amax_tensor.item())

    # Double-quant scale: amax / 6.0 (FP4 max) / 448.0 (FP8 max)
    if amax and amax != 0:
        dq_scale_value = amax / 6.0 / 448.0
    else:
        dq_scale_value = 1.0

    pre_nodes = []
    post_nodes = []

    if needs_reshape:
        input_3d, output_3d, reshape_nodes, reshape_inits = _reshape_for_dynquant(
            graph, input_name, output_name, node.name, existing_names
        )
        for init in reshape_inits:
            graph.initializer.append(init)
        pre_nodes = reshape_nodes[:5]
        post_nodes = reshape_nodes[5:]
        working_input = input_3d
        working_output = output_3d
    else:
        working_input = input_name
        working_output = output_name

    quant_output_name = f"{node.name}_quantized"
    scale_output_name = f"{node.name}_scale"
    scale_dq_output_name = f"{node.name}_scale_dq"
    dq_scale_name = f"{node.name}_dq_scale"
    dq_float_output = f"{node.name}_dq_float"

    graph.node.remove(node)

    # Initializer for double-quant scale (FP32)
    dq_scale_init = onnx.numpy_helper.from_array(
        np.array(dq_scale_value, dtype=np.float32), dq_scale_name
    )
    graph.initializer.append(dq_scale_init)

    # TRT_FP4DynamicQuantize: input -> (FP4 data, FP8 scale)
    dynquant_node = onnx.helper.make_node(
        "TRT_FP4DynamicQuantize",
        inputs=[working_input, dq_scale_name],
        outputs=[quant_output_name, scale_output_name],
        name=f"{node.name}_DynamicQuantize",
        domain="trt",
        axis=-1,
        block_size=block_size,
        scale_type=onnx.TensorProto.FLOAT8E4M3FN,
    )

    # DQ1: scale FP8 -> FP32 (trt domain, no axis/block_size)
    scale_dequant_node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[scale_output_name, dq_scale_name],
        outputs=[scale_dq_output_name],
        name=f"{node.name}_DequantizeLinear_scale",
        domain="trt",
    )

    # DQ2: data FP4 -> FP32 (trt domain, with axis/block_size)
    data_dequant_node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[quant_output_name, scale_dq_output_name],
        outputs=[dq_float_output],
        name=f"{node.name}_DequantizeLinear_data",
        domain="trt",
        axis=-1,
        block_size=block_size,
    )

    # Cast FP32 -> original type (BF16)
    cast_node = onnx.helper.make_node(
        "Cast",
        inputs=[dq_float_output],
        outputs=[working_output],
        name=f"{node.name}_Cast_to_original",
        to=original_output_type,
    )

    # Value infos for intermediates
    graph.value_info.append(
        onnx.helper.make_tensor_value_info(quant_output_name, onnx.TensorProto.FLOAT4E2M1, None)
    )
    graph.value_info.append(
        onnx.helper.make_tensor_value_info(scale_output_name, onnx.TensorProto.FLOAT8E4M3FN, None)
    )
    graph.value_info.append(
        onnx.helper.make_tensor_value_info(scale_dq_output_name, onnx.TensorProto.FLOAT, None)
    )
    graph.value_info.append(
        onnx.helper.make_tensor_value_info(dq_float_output, onnx.TensorProto.FLOAT, None)
    )

    # Output maintains original type
    if working_output in value_info_map:
        value_info_map[working_output].type.tensor_type.elem_type = original_output_type
    else:
        graph.value_info.append(
            onnx.helper.make_tensor_value_info(working_output, original_output_type, None)
        )

    existing_names.update(
        [
            dynquant_node.name,
            scale_dequant_node.name,
            data_dequant_node.name,
            cast_node.name,
            quant_output_name,
            scale_output_name,
            scale_dq_output_name,
            dq_scale_name,
            dq_float_output,
        ]
    )

    graph.node.extend(
        pre_nodes + [dynquant_node, scale_dequant_node, data_dequant_node, cast_node] + post_nodes
    )


class NVFP4QuantExporter(ONNXQuantExporter):
    """Exporter for NVFP4 quantization."""

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model for NVFP4 quantization."""
        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for weights in the ONNX model."""
        logger.info("Computing scales for NVFP4 quantization")

        graph = onnx_model.graph
        initializers = graph.initializer
        initializer_indices = {init.name: idx for idx, init in enumerate(initializers)}

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        weight_count = 0
        activation_count = 0

        for node in fp4_qdq_nodes:
            idx = initializer_indices.get(node.input[0], None)
            if idx is None:
                activation_count += 1
                continue

            weight_count += 1
            tensor = initializers[idx]
            w32 = utils.read_f16_tensor_as_fp32(tensor)

            block_size_attr = next((a for a in node.attribute if a.name == "block_size"), None)
            if not block_size_attr:
                logger.error(f"block_size attribute not found for node {node.name}")
                continue
            block_size = block_size_attr.i

            sw_f32_per_tensor = get_weights_scaling_factor_2(w32)
            sw_f32_per_block = get_weights_scaling_factor(w32, block_size, sw_f32_per_tensor)

            logger.debug(f"Computed scales for weight {node.input[0]} with block size {block_size}")

            attr = node.attribute.add()
            attr.name = "_sw_f32_per_tensor"
            attr.floats.extend(sw_f32_per_tensor.flatten().tolist())

            attr = node.attribute.add()
            attr.name = "_sw_f32_per_block"
            attr.floats.extend(sw_f32_per_block.flatten().tolist())

            attr = node.attribute.add()
            attr.name = "_sw_f32_per_block_shape"
            attr.ints.extend(sw_f32_per_block.shape)

        logger.info(
            f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes: {weight_count} weight, {activation_count} activation"
        )
        return onnx_model

    @staticmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses weights to FP4 format."""
        logger.info("Compressing weights for NVFP4 quantization")

        graph = onnx_model.graph
        initializers = graph.initializer
        initializer_indices = {init.name: idx for idx, init in enumerate(initializers)}

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        compressed_count = 0

        for node in fp4_qdq_nodes:
            idx = initializer_indices.get(node.input[0], None)
            if idx is None:
                continue

            compressed_count += 1
            tensor = initializers[idx]
            w32 = utils.read_f16_tensor_as_fp32(tensor)

            block_size_attr = next((a for a in node.attribute if a.name == "block_size"), None)
            if not block_size_attr:
                logger.error(f"block_size attribute not found for node {node.name}")
                continue
            block_size = block_size_attr.i

            sw_f32_per_tensor = None
            sw_f32_per_block = None
            sw_per_block_shape = None

            for attr in node.attribute:
                if attr.name == "_sw_f32_per_tensor":
                    sw_f32_per_tensor = np.array(list(attr.floats), dtype=np.float32)
                elif attr.name == "_sw_f32_per_block":
                    sw_f32_per_block = np.array(list(attr.floats), dtype=np.float32)
                elif attr.name == "_sw_f32_per_block_shape":
                    sw_per_block_shape = tuple(attr.ints)

            if sw_f32_per_tensor is None or sw_f32_per_block is None or sw_per_block_shape is None:
                logger.error(f"Scales not found for {node.input[0]}")
                continue

            sw_f32_per_block = sw_f32_per_block.reshape(sw_per_block_shape)
            w_f32 = quantize(w32, block_size, sw_f32_per_block, sw_f32_per_tensor)

            try:
                w_f4 = _cast_fp4(w_f32)
                sw_f8_per_block = _cast_fp8(sw_f32_per_block)
            except Exception as e:
                logger.error(f"Failed to cast weights for {node.input[0]}: {e}")
                continue

            attr = node.attribute.add()
            attr.name = "_w_f4"
            attr.t.CopyFrom(numpy_helper.from_array(w_f4, "w_f4"))

            attr = node.attribute.add()
            attr.name = "_sw_f8_per_block"
            attr.t.CopyFrom(numpy_helper.from_array(sw_f8_per_block, "sw_f8"))

            logger.debug(f"Compressed weight {node.input[0]} to FP4")

        logger.info(f"Compressed {compressed_count} weight tensors to FP4")
        return onnx_model

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model, replacing FP4QDQ nodes with TRT ops."""
        logger.info("Post-processing NVFP4 quantization")

        graph = onnx_model.graph

        for node in graph.node:
            if node.op_type == "DequantizeLinear" and node.domain == "":
                # Check if it has block_size (FP4 pattern)
                if any(a.name == "block_size" for a in node.attribute):
                    logger.warn(f"fixing domain on {node.name}")
                    node.domain = "trt"

        initializers_to_delete = set()
        tensor_consumers = get_tensor_consumer_nodes(graph)

        initializer_indices = {init.name: idx for idx, init in enumerate(graph.initializer)}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        graph_inputs = {inp.name for inp in graph.input}
        existing_names = _collect_existing_names(graph)
        cast_cache = {}

        def _get_precision_dtype() -> str:
            for init in graph.initializer:
                if init.data_type == onnx.TensorProto.BFLOAT16:
                    return "BFloat16"
            return "Half"

        def _cast_input_dtypes(node: onnx.NodeProto, precision_dtype: str):
            if node.op_type == "Transpose":
                consumers = tensor_consumers.get(node.output[0], [])
                if consumers and consumers[0].op_type == "MatMul":
                    node = consumers[0]

            for i, input_name in enumerate(node.input[:2]):
                cache_key = f"{input_name}_{precision_dtype}"
                if cache_key in cast_cache:
                    node.input[i] = cast_cache[cache_key]
                    continue

                cast_output_name = f"{node.name}_input{i}_cast_{precision_dtype}"
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_output_name],
                    to=onnx_dtype_map[precision_dtype],
                    name=f"{node.name}_Cast_input{i}",
                )

                graph.node.append(cast_node)
                existing_names.add(cast_node.name)
                existing_names.add(cast_output_name)
                node.input[i] = cast_output_name
                cast_cache[cache_key] = cast_output_name

        precision_dtype = _get_precision_dtype()
        logger.debug(f"Using precision dtype: {precision_dtype}")

        fp4_qdq_nodes = [
            (idx, node) for idx, node in enumerate(graph.node) if node.op_type == "TRT_FP4QDQ"
        ]
        weight_count = 0
        activation_count = 0

        for node_idx, node in reversed(fp4_qdq_nodes):
            block_size_attr = next((a for a in node.attribute if a.name == "block_size"), None)
            if not block_size_attr:
                logger.error(f"block_size attribute not found for node {node.name}")
                continue
            block_size = block_size_attr.i

            idx = initializer_indices.get(node.input[0], None)
            if idx is None:
                # Activation quantization
                activation_count += 1
                logger.debug(
                    f"Replacing activation FP4QDQ node {node.name} with TRT_FP4DynamicQuantize"
                )
                _replace_activation_fp4qdq_with_dynquant(
                    graph, node, node_idx, value_info_map, existing_names, block_size
                )
                continue

            # Weight quantization
            weight_count += 1
            initializers_to_delete.add(graph.initializer[idx].name)

            w_f4 = None
            sw_f8_per_block = None
            sw_f32_per_tensor = None

            for attr in node.attribute:
                if attr.name == "_w_f4":
                    w_f4 = numpy_helper.to_array(attr.t)
                elif attr.name == "_sw_f8_per_block":
                    sw_f8_per_block = numpy_helper.to_array(attr.t)
                elif attr.name == "_sw_f32_per_tensor":
                    sw_f32_per_tensor = np.array(list(attr.floats), dtype=np.float32)

            if w_f4 is None or sw_f8_per_block is None or sw_f32_per_tensor is None:
                logger.error(f"Compressed data not found for {node.input[0]}")
                continue

            logger.debug(f"Replacing weight FP4QDQ node {node.name} with 2 DQ nodes")

            _replace_weight_fp4qdq_with_2dq(
                graph,
                node,
                node_idx,
                initializer_indices,
                value_info_map,
                graph_inputs,
                existing_names,
                w_f4,
                sw_f32_per_tensor,
                sw_f8_per_block,
                block_size,
            )

            consumers = tensor_consumers.get(node.output[0], [])
            if consumers:
                _cast_input_dtypes(consumers[0], precision_dtype)

        # Remove old initializers
        new_initializers = [
            init for init in graph.initializer if init.name not in initializers_to_delete
        ]

        graph.ClearField("initializer")
        graph.initializer.extend(new_initializers)

        logger.info(
            f"Processed {weight_count} weight and {activation_count} activation quantization nodes"
        )
        logger.info(f"Removed {len(initializers_to_delete)} initializers")

        return onnx_model
