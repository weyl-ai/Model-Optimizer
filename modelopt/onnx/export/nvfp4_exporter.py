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

"""NVFP4 quantization exporter."""

import numpy as np
import onnx
import torch
import time
import psutil
import os
from onnx import numpy_helper
from typing import Dict, Any

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


def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_gb": mem_info.rss / (1024**3),
        "vms_gb": mem_info.vms / (1024**3),
        "percent": process.memory_percent(),
        "available_gb": psutil.virtual_memory().available / (1024**3),
    }


def log_memory_and_size(operation: str, tensor_name: str = None, tensor: np.ndarray = None):
    """Log memory usage and optionally tensor size."""
    mem = get_memory_info()
    msg = f"[{operation}] Memory: {mem['rss_gb']:.2f}GB RSS, {mem['percent']:.1f}% used, {mem['available_gb']:.2f}GB available"
    if tensor is not None and tensor_name:
        tensor_size_mb = tensor.nbytes / (1024**2)
        msg += f" | Tensor '{tensor_name}': shape={tensor.shape}, size={tensor_size_mb:.2f}MB"
    logger.info(msg)


def _cast_fp4(array: np.ndarray) -> np.ndarray:
    """Cast a numpy array to FLOAT4E2M1 using PyTorch."""
    array_f32_t = torch.from_numpy(array)
    array_f32_t_shape = array_f32_t.shape
    assert array_f32_t_shape[0] % 2 == 0, "array_f32_t_shape[0] must be divisible by 2"
    array_f4_t_shape = (array_f32_t_shape[0] // 2, *array_f32_t_shape[1:])

    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()

    array_f4_t = NVFP4QTensor._cast_fp4(array_f32_t)
    array_f4_t = array_f4_t.flatten()
    array_f4_t_packed = (array_f4_t[::2] | (array_f4_t[1::2] << 4)).reshape(array_f4_t_shape)
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


def _replace_fp4qdq_with_2dq(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    initializer_indices: dict[str, int],
    value_info_map: dict[str, onnx.ValueInfoProto],
    graph_inputs: set[str],
    w_f4: np.ndarray,
    sw_f32_per_tensor: np.ndarray,
    sw_f8_per_block: np.ndarray,
    block_size: int,
):
    """Replaces the given node with two DequantizeLinear nodes."""

    def _add_initializer(initializer):
        if initializer.name not in initializer_indices:
            graph.initializer.append(initializer)

    def _add_input_value_info(graph, tensor_proto):
        assert tensor_proto.name not in graph_inputs, (
            f"{tensor_proto.name} already in graph inputs."
        )
        assert tensor_proto.name not in value_info_map, (
            f"{tensor_proto.name} already in value info."
        )
        value_info = onnx.helper.make_tensor_value_info(
            tensor_proto.name, tensor_proto.data_type, tensor_proto.dims
        )
        graph.input.append(value_info)

    weight_name = node.input[0]

    # Generate unique names for the initializers
    w_f4_name = weight_name + "_f4"
    sw_f8_per_block_name = weight_name + "_f8_scale"
    sw_f32_per_tensor_name = sw_f8_per_block_name + "_f32_scale"

    # Create TensorProto for initializers
    w_f4_proto = onnx.helper.make_tensor(
        name=w_f4_name,
        data_type=onnx_dtype_map["Float4"],
        dims=[w_f4.shape[0] * 2, *w_f4.shape[1:]],
        vals=w_f4.tobytes(),
        raw=True,
    )
    sw_f32_per_tensor_proto = onnx.numpy_helper.from_array(
        sw_f32_per_tensor, sw_f32_per_tensor_name
    )
    sw_f8_per_block_proto = onnx.helper.make_tensor(
        name=sw_f8_per_block_name,
        data_type=onnx_dtype_map["Float8"],
        dims=[*sw_f8_per_block.shape],
        vals=sw_f8_per_block.tobytes(),
        raw=True,
    )

    # Add ValueInfo for the initializers
    _add_input_value_info(graph, w_f4_proto)
    _add_input_value_info(graph, sw_f32_per_tensor_proto)
    _add_input_value_info(graph, sw_f8_per_block_proto)

    # Add the initializers to the graph
    _add_initializer(w_f4_proto)
    _add_initializer(sw_f32_per_tensor_proto)
    _add_initializer(sw_f8_per_block_proto)

    # Create DequantizeLinear_1: (sw_f8_per_block, sw_f32_per_tensor) -> sw_f32
    sw_f32_name = weight_name + "_f32_scale"
    dequant1 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[sw_f8_per_block_proto.name, sw_f32_per_tensor_proto.name],
        outputs=[sw_f32_name],
        name=weight_name + "_DequantizeLinear",
    )

    # Create DequantizeLinear_2: (w_f4, sw_f32) -> w_32
    w32_name = node.output[0]
    dequant2 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[w_f4_proto.name, sw_f32_name],
        outputs=[w32_name],
        name=weight_name + "_DequantizeLinear_1",
        axis=-1,
        block_size=block_size,
    )

    # Add value_info for sw_f32
    sw_f32_type_proto = onnx.helper.make_tensor_type_proto(
        elem_type=onnx_dtype_map["Float"], shape=sw_f8_per_block.shape
    )
    sw_f16_value_info = onnx.helper.make_value_info(name=sw_f32_name, type_proto=sw_f32_type_proto)
    graph.value_info.append(sw_f16_value_info)

    # Change the data type of output
    if w32_name in value_info_map:
        value_info_map[w32_name].type.tensor_type.elem_type = onnx_dtype_map["Float"]
    else:
        raise ValueError(f"ValueInfo for {w32_name} not found.")

    # Add the new nodes to the graph
    graph.node.extend([dequant1, dequant2])


class NVFP4QuantExporter(ONNXQuantExporter):
    """Exporter for NVFP4 quantization."""

    @classmethod
    def process_model(cls, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Processes the ONNX model with comprehensive logging."""
        overall_start = time.time()
        logger.info("#" * 100)
        logger.info("STARTING NVFP4 QUANTIZATION EXPORT PROCESS")
        logger.info("#" * 100)
        log_memory_and_size("OVERALL START")

        total_nodes = len(onnx_model.graph.node)
        total_initializers = len(onnx_model.graph.initializer)
        fp4_nodes = sum(1 for node in onnx_model.graph.node if node.op_type == "TRT_FP4QDQ")

        logger.info(
            f"Model Stats: {total_nodes} nodes, {total_initializers} initializers, {fp4_nodes} FP4QDQ nodes"
        )

        onnx_model = cls.pre_process(onnx_model)
        onnx_model = cls.compute_scales(onnx_model)
        onnx_model = cls.compress_weights(onnx_model)
        onnx_model = cls.post_process(onnx_model)

        final_nodes = len(onnx_model.graph.node)
        final_initializers = len(onnx_model.graph.initializer)
        elapsed_total = time.time() - overall_start

        logger.info("#" * 100)
        logger.info(f"COMPLETED NVFP4 QUANTIZATION EXPORT in {elapsed_total:.2f}s")
        logger.info(
            f"Final Stats: {final_nodes} nodes ({final_nodes - total_nodes:+d}), {final_initializers} initializers ({final_initializers - total_initializers:+d})"
        )
        log_memory_and_size("OVERALL END")
        logger.info("#" * 100)

        return onnx_model

    @staticmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model. No-op for NVFP4."""
        return onnx_model

    @staticmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes scales for weights. Stores as node attributes."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("STARTING: Computing scales for NVFP4 quantization")
        log_memory_and_size("compute_scales START")

        graph = onnx_model.graph
        initializers = graph.initializer
        initializer_indices = {init.name: idx for idx, init in enumerate(initializers)}

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        logger.info(f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes to process")

        processed_count = 0
        skipped_count = 0
        total_tensor_size_mb = 0

        for i, node in enumerate(fp4_qdq_nodes):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(fp4_qdq_nodes) - i) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {i}/{len(fp4_qdq_nodes)} nodes, {elapsed:.1f}s elapsed, ETA: {eta:.1f}s"
                )
                log_memory_and_size(f"compute_scales node {i}")

            idx = initializer_indices.get(node.input[0], None)
            if idx is None:
                logger.debug(
                    f"Skipping node {node.name} - input '{node.input[0]}' is not an initializer"
                )
                skipped_count += 1
                continue

            tensor = initializers[idx]
            tensor_size_mb = len(tensor.raw_data) / (1024**2) if tensor.raw_data else 0
            total_tensor_size_mb += tensor_size_mb

            logger.debug(f"Processing weight '{node.input[0]}' ({tensor_size_mb:.2f}MB)")
            w32 = utils.read_f16_tensor_as_fp32(tensor)

            # Compute scales
            sw_f32_per_tensor = get_weights_scaling_factor_2(w32)
            block_size = node.attribute[0].i
            sw_f32_per_block = get_weights_scaling_factor(w32, block_size, sw_f32_per_tensor)

            logger.debug(f"Computed scales for weight {node.input[0]} with block size {block_size}")
            processed_count += 1

            # Store scales as node attributes
            sw_per_tensor_attr = node.attribute.add()
            sw_per_tensor_attr.name = "_sw_f32_per_tensor"
            sw_per_tensor_attr.floats.extend(sw_f32_per_tensor.flatten().tolist())

            sw_per_block_attr = node.attribute.add()
            sw_per_block_attr.name = "_sw_f32_per_block"
            sw_per_block_attr.floats.extend(sw_f32_per_block.flatten().tolist())

            sw_per_block_shape_attr = node.attribute.add()
            sw_per_block_shape_attr.name = "_sw_f32_per_block_shape"
            sw_per_block_shape_attr.ints.extend(sw_f32_per_block.shape)

        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"COMPLETED: compute_scales in {elapsed_time:.2f}s")
        logger.info(
            f"Processed {processed_count} weight quantizer nodes, skipped {skipped_count} activation quantizer nodes"
        )
        logger.info(f"Total weight tensor size processed: {total_tensor_size_mb:.2f}MB")
        log_memory_and_size("compute_scales END")
        logger.info("=" * 80)

        return onnx_model

    @staticmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses weights to FP4 format and scales to FP8 format."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("STARTING: Compressing weights for NVFP4 quantization")
        log_memory_and_size("compress_weights START")

        graph = onnx_model.graph
        initializers = graph.initializer
        initializer_indices = {init.name: idx for idx, init in enumerate(initializers)}

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        logger.info(f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes for weight compression")

        processed_count = 0
        skipped_count = 0
        total_compressed_size_mb = 0

        for i, node in enumerate(fp4_qdq_nodes):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(fp4_qdq_nodes) - i) / rate if rate > 0 else 0
                logger.info(
                    f"Compression Progress: {i}/{len(fp4_qdq_nodes)} nodes, {elapsed:.1f}s elapsed, ETA: {eta:.1f}s"
                )
                log_memory_and_size(f"compress_weights node {i}")

            idx = initializer_indices.get(node.input[0], None)
            if idx is None:
                logger.debug(
                    f"Skipping node {node.name} - input '{node.input[0]}' is not an initializer"
                )
                skipped_count += 1
                continue

            tensor = initializers[idx]
            original_size_mb = len(tensor.raw_data) / (1024**2) if tensor.raw_data else 0
            logger.debug(f"Compressing weight '{node.input[0]}' ({original_size_mb:.2f}MB)")

            w32 = utils.read_f16_tensor_as_fp32(tensor)
            block_size = node.attribute[0].i

            # Retrieve scales from node attributes
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

            assert sw_f32_per_tensor is not None, f"Scales not found for {node.input[0]}"
            assert sw_f32_per_block is not None, f"Block scales not found for {node.input[0]}"
            assert sw_per_block_shape is not None, (
                f"Block scale shape not found for {node.input[0]}"
            )

            sw_f32_per_block = sw_f32_per_block.reshape(sw_per_block_shape)

            # Quantize weights
            w_f32 = quantize(w32, block_size, sw_f32_per_block, sw_f32_per_tensor)

            # Cast to FP4 and FP8
            w_f4 = _cast_fp4(w_f32)
            sw_f8_per_block = _cast_fp8(sw_f32_per_block)

            compressed_size_mb = w_f4.nbytes / (1024**2)
            total_compressed_size_mb += compressed_size_mb
            compression_ratio = (
                original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
            )
            logger.debug(
                f"Compressed '{node.input[0]}': {original_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB (ratio: {compression_ratio:.2f}x)"
            )

            # Store compressed data as node attributes
            w_f4_attr = node.attribute.add()
            w_f4_attr.name = "_w_f4"
            w_f4_attr.t.CopyFrom(numpy_helper.from_array(w_f4, "w_f4"))

            sw_f8_attr = node.attribute.add()
            sw_f8_attr.name = "_sw_f8_per_block"
            sw_f8_attr.t.CopyFrom(numpy_helper.from_array(sw_f8_per_block, "sw_f8"))

            logger.debug(f"Compressed weight {node.input[0]} to FP4")
            processed_count += 1

        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"COMPLETED: compress_weights in {elapsed_time:.2f}s")
        logger.info(
            f"Compressed {processed_count} weight tensors, skipped {skipped_count} activation quantizer nodes"
        )
        logger.info(f"Total compressed size: {total_compressed_size_mb:.2f}MB")
        log_memory_and_size("compress_weights END")
        logger.info("=" * 80)

        return onnx_model

    @staticmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Replaces TRT_FP4QDQ nodes with DequantizeLinear nodes."""
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("STARTING: Post-processing NVFP4 quantization")
        log_memory_and_size("post_process START")

        graph = onnx_model.graph
        initializers_to_delete = set()
        tensor_consumers = get_tensor_consumer_nodes(graph)
        initializer_indices = {init.name: idx for idx, init in enumerate(graph.initializer)}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        graph_inputs = {inp.name for inp in graph.input}

        def _get_precision_dtype() -> str:
            precision_dtype = "Half"
            for initializer in graph.initializer:
                if initializer.data_type == 16:
                    precision_dtype = "BFloat16"
                    break
            return precision_dtype

        def _cast_input_dtypes(node: onnx.NodeProto, precision_dtype: str):
            if node.op_type == "Transpose":
                maybe_matmul = tensor_consumers[node.output[0]][0]
                assert maybe_matmul.op_type == "MatMul"
                node = maybe_matmul

            for i, input_name in enumerate(node.input[:2]):
                cast_output_name = node.name + "_" + input_name + "_f16"
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_output_name],
                    to=onnx_dtype_map[precision_dtype],
                )
                graph.node.extend([cast_node])
                node.input[i] = cast_output_name

        precision_dtype = _get_precision_dtype()
        logger.debug(f"Using precision dtype: {precision_dtype}")

        fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
        logger.info(f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes to convert to DequantizeLinear")

        processed_count = 0
        skipped_count = 0
        nodes_created = 0

        for i, node in enumerate(fp4_qdq_nodes):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(fp4_qdq_nodes) - i) / rate if rate > 0 else 0
                logger.info(
                    f"Post-process Progress: {i}/{len(fp4_qdq_nodes)} nodes, {elapsed:.1f}s elapsed, ETA: {eta:.1f}s"
                )
                log_memory_and_size(f"post_process node {i}")

            idx = initializer_indices.get(node.input[0], None)

            if idx is None:
                # Activation quantizer: wire input directly to consumers
                for consumer in tensor_consumers.get(node.output[0], []):
                    for j, inp in enumerate(consumer.input):
                        if inp == node.output[0]:
                            consumer.input[j] = node.input[0]
                skipped_count += 1
                continue

            initializers_to_delete.add(graph.initializer[idx].name)

            # Retrieve compressed data from node attributes
            block_size = node.attribute[0].i
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

            assert w_f4 is not None, f"Compressed weights not found for {node.input[0]}"
            assert sw_f8_per_block is not None, f"FP8 scales not found for {node.input[0]}"
            assert sw_f32_per_tensor is not None, f"Per-tensor scales not found for {node.input[0]}"

            logger.debug(f"Converting FP4QDQ node '{node.input[0]}' to DequantizeLinear nodes")

            _replace_fp4qdq_with_2dq(
                graph,
                node,
                initializer_indices,
                value_info_map,
                graph_inputs,
                w_f4,
                sw_f32_per_tensor,
                sw_f8_per_block,
                block_size,
            )
            nodes_created += 2

            # Cast input dtypes for the next node
            next_node = tensor_consumers[node.output[0]][0]
            _cast_input_dtypes(next_node, precision_dtype)
            processed_count += 1

        # Remove old initializers
        logger.info(f"Removing {len(initializers_to_delete)} old initializers")
        new_initializers = [
            init for init in graph.initializer if init.name not in initializers_to_delete
        ]
        graph.ClearField("initializer")
        graph.initializer.extend(new_initializers)

        # Batch remove all FP4QDQ nodes
        nodes_to_remove = {id(n) for n in fp4_qdq_nodes}
        new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
        graph.ClearField("node")
        graph.node.extend(new_nodes)

        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"COMPLETED: post_process in {elapsed_time:.2f}s")
        logger.info(
            f"Processed {processed_count} weight quantizer nodes, skipped {skipped_count} activation quantizer nodes"
        )
        logger.info(
            f"Created {nodes_created} DequantizeLinear nodes, removed {len(initializers_to_delete)} initializers"
        )
        log_memory_and_size("post_process END")
        logger.info("=" * 80)

        return onnx_model
