# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "vllm",
# ]
# ///
"""
vLLM OpenAI-compatible server for the open-weight models used in the paper.

Serves one model at a time; run one process per model on its own GPUs/port.

Examples:
  python serve.py --model Qwen/Qwen3.5-35B-A3B      --port 8000 --gpus 4,5,6,7 --tp 4
  python serve.py --model Qwen/Qwen3.5-9B           --port 8001 --gpus 0,1     --tp 2
  python serve.py --model Qwen/Qwen3.5-4B           --port 8002 --gpus 3       --tp 1
  python serve.py --model meta-llama/Llama-3.1-8B-Instruct  --port 8004 --gpus 0 --tp 1
  python serve.py --model meta-llama/Llama-3.3-70B-Instruct --port 8000 --gpus 4,5,6,7 --tp 4
  python serve.py --model google/gemma-4-26B-A4B-it --port 8000 --gpus 4,5,6,7 --tp 4
"""

import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True, help="HuggingFace model id to serve")
ap.add_argument("--port", default="8000")
ap.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0,1")
ap.add_argument("--tp", default="1", help="tensor-parallel-size")
ap.add_argument("--max-model-len", default="8192")
ap.add_argument("--gpu-memory-utilization", default="0.92")
args = ap.parse_args()

# Use the current working directory as the HF cache root.
CACHE_DIR = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", args.model,
    "--host", "0.0.0.0",
    "--port", args.port,
    "--tensor-parallel-size", args.tp,
    "--gpu-memory-utilization", args.gpu_memory_utilization,
    "--max-model-len", args.max_model_len,
    "--max-num-batched-tokens", args.max_model_len,
    "--max-num-seqs", "256",
]

print(f"Using HF cache at: {CACHE_DIR}")
print(f"Serving {args.model} on port {args.port} (gpus={args.gpus}, tp={args.tp})")
subprocess.run(cmd)
