#!/usr/bin/env python3
"""Convert a trained WCAG Laya model to ONNX for browser deployment.

    python scripts/build_wcag_model.py              # Use default build/wcag
    python scripts/build_wcag_model.py --out build/wcag-v2
    python scripts/build_wcag_model.py --variants qdq8  # Only closest-to-original

This converts the fine-tuned WCAG model (421M params) to ONNX Runtime Web format,
quantizes it, and splits into browser-sized parts for GitHub Pages deployment.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import laya


def convert_checkpoint(checkpoint_path, output_dir):
    """Convert Laya checkpoint to ONNX."""
    print(f"Converting checkpoint: {checkpoint_path}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = laya.load(checkpoint_path, device=device)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    
    # Laya uses specific export method
    agent.model.export(
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        opset_version=14
    )
    
    print(f"ONNX model saved to {onnx_path}")
    
    return onnx_path


def quantize_model(onnx_path, output_dir, variant="qdq8"):
    """Quantize the ONNX model."""
    print(f"Quantizing model with variant: {variant}")
    
    import onnx
    from onnxconverter_common import float16
    
    model = onnx.load(onnx_path)
    
    if variant == "q8e8":
        # Block-128 quantization (more aggressive)
        from onnxruntime.tools.quantization import quantize_dynamic
        quantized = quantize_dynamic(
            model,
            {onnx.helper.make_tensor_value_info('input_ids', onnx.TensorProto.INT64, [1, 512])},
            per_channel=False
        )
    elif variant == "q4e8":
        # QAT-style quantization
        from onnxruntime.tools.quantization import quantize_dynamic
        quantized = quantize_dynamic(
            model,
            {onnx.helper.make_tensor_value_info('input_ids', onnx.TensorProto.INT64, [1, 512])},
            per_channel=False
        )
    else:  # qdq8 - default
        # Per-channel quantization
        from onnxruntime.tools.quantization import quantize_dynamic
        quantized = quantize_dynamic(
            model,
            {onnx.helper.make_tensor_value_info('input_ids', onnx.TensorProto.INT64, [1, 512])},
            per_channel=True
        )
    
    quantized_path = os.path.join(output_dir, f"model_{variant}.onnx")
    onnx.save(quantized, quantized_path)
    print(f"Quantized model saved to {quantized_path}")
    
    return quantized_path


def split_model(model_path, output_dir, part_size_mb=24):
    """Split model into parts for GitHub Pages."""
    print(f"Splitting model: {model_path}")
    
    file_size = os.path.getsize(model_path)
    part_size = part_size_mb * 1024 * 1024
    
    if file_size <= part_size:
        # Single file
        dest = os.path.join(output_dir, "model.onnx")
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        shutil.copy(model_path, dest)
        print(f"Model fits in single part ({file_size / (1024*1024):.1f} MB)")
        return
    
    # Split into parts
    with open(model_path, "rb") as f:
        model_data = f.read()
    
    num_parts = (len(model_data) + part_size - 1) // part_size
    
    with open(model_path, "rb") as f:
        for i in range(num_parts):
            start = i * part_size
            end = min((i + 1) * part_size, len(model_data))
            part_data = model_data[start:end]
            
            part_filename = f"model-{i+1:02d}.onnx"
            part_path = os.path.join(output_dir, part_filename)
            
            with open(part_path, "wb") as pf:
                pf.write(part_data)
            
            print(f"  Part {i+1}/{num_parts}: {part_filename} ({len(part_data) / (1024*1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert WCAG Laya model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--out", default="build/wcag", help="Output directory")
    parser.add_argument("--variants", default="qdq8", choices=["qdq8", "q8e8", "q4e8"], help="Quantization variant")
    parser.add_argument("--split", action="store_true", help="Split into parts for GitHub Pages")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    
    # Convert
    onnx_path = convert_checkpoint(args.checkpoint, args.out)
    
    # Quantize
    quantized_path = quantize_model(onnx_path, args.out, variant=args.variants)
    
    # Split if needed
    if args.split:
        split_model(quantized_path, os.path.join(args.out, "model"))
    
    print(f"\nDone! Model ready for deployment.")


if __name__ == "__main__":
    main()
