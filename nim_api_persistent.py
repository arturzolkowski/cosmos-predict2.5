#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Cosmos Predict2.5 — Persistent Benchmark API

FastAPI-based inference server for performance benchmarking of Cosmos Predict2.5.
Compatible with the benchmarking-speed client used in JET CI.

Architecture:
  - Rank 0 runs the FastAPI/uvicorn HTTP server and handles request parsing.
  - Ranks 1…N run a worker loop, receiving inference jobs via NCCL broadcast.
  - All ranks participate in each forward pass (context parallelism).
  - The model is loaded once at startup via SetupArguments.

Model selection:
  Set MODEL_NAME env var to any base-variant model key (default: 2B/post-trained):
    2B/pre-trained, 2B/post-trained, 2B/distilled, 14B/pre-trained, 14B/post-trained

Endpoints:
  POST /v1/infer         — run inference
  GET  /v1/health/live   — liveness probe (always 200)
  GET  /v1/health/ready  — readiness probe (200 once model loaded)
  GET  /health           — uptime, request count, GPU count

Run (single GPU):
  python3 nim_api_persistent.py

Run (multi-GPU, e.g. 4 GPUs):
  torchrun --nproc_per_node=4 nim_api_persistent.py
"""

import base64
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import requests
import torch
import torch.distributed as dist

# Deferred imports — loaded after distributed init to avoid CUDA init on workers
Inference = None
InferenceArguments = None
SetupArguments = None
InferenceType = None

# ── Global state ──────────────────────────────────────────────────────────────

INFERENCE_ENGINE = None
ENGINE_LOCK = threading.Lock()
STARTUP_TIME = time.time()
REQUEST_COUNT = 0

# Valid base-variant model keys (SetupArguments only supports BASE variant)
VALID_MODELS = [
    "2B/pre-trained",
    "2B/post-trained",
    "2B/distilled",
    "14B/pre-trained",
    "14B/post-trained",
]
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "2B/post-trained")


# ── Logging ───────────────────────────────────────────────────────────────────


def log(msg: str, rank: int = None, level: str = "INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    r = rank if rank is not None else (get_rank() if dist.is_initialized() else 0)
    print(f"[{ts}|{level}|Rank {r}] {msg}", flush=True)


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


# ── GPU diagnostics ───────────────────────────────────────────────────────────


def log_gpu_info():
    try:
        count = torch.cuda.device_count()
        log(f"CUDA available: {torch.cuda.is_available()}, GPU count: {count}")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            log(f"  GPU {i}: {name}, {mem:.1f} GB")
    except Exception as e:
        log(f"Could not get GPU info: {e}", level="WARN")


def log_memory_usage(label: str = ""):
    try:
        rank = get_rank()
        alloc = torch.cuda.memory_allocated(rank) / (1024**3)
        res = torch.cuda.memory_reserved(rank) / (1024**3)
        log(f"GPU memory {label}: allocated={alloc:.2f}GB, reserved={res:.2f}GB")
    except Exception:
        pass


# ── Environment ───────────────────────────────────────────────────────────────


def apply_nim_env_settings():
    settings = {
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/.cache/",
        "TORCH_NCCL_USE_COMM_NONBLOCKING": "0",
        "NCCL_CUMEM_HOST_ENABLE": "0",
    }
    for k, v in settings.items():
        os.environ.setdefault(k, v)
    if is_rank0():
        log("Environment settings applied:")
        for k in settings:
            log(f"  {k}={os.environ.get(k)}")


# ── Model loading ─────────────────────────────────────────────────────────────


def load_inference_engine(model_name: str = None):
    """Load the Inference engine once on all ranks."""
    global INFERENCE_ENGINE, Inference, InferenceArguments, SetupArguments, InferenceType

    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in VALID_MODELS:
        log(
            f"WARNING: '{model_name}' is not in the known base-variant list {VALID_MODELS}. "
            "Proceeding anyway — SetupArguments will validate.",
            level="WARN",
        )

    load_start = time.time()
    log(f"Importing cosmos_predict2 modules...")

    try:
        from cosmos_predict2.config import InferenceArguments as IA
        from cosmos_predict2.config import InferenceType as IT
        from cosmos_predict2.config import SetupArguments as SA
        from cosmos_predict2.inference import Inference as Inf

        Inference = Inf
        InferenceArguments = IA
        SetupArguments = SA
        InferenceType = IT
        log(f"Modules imported in {time.time() - load_start:.1f}s")
    except Exception as e:
        log(f"FATAL: Failed to import cosmos_predict2 modules: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        raise

    log(f"Creating inference engine: model={model_name}, world_size={get_world_size()}")

    try:
        setup_args = SetupArguments(
            model=model_name,
            context_parallel_size=get_world_size(),
            disable_guardrails=True,
            output_dir=Path("/tmp/cosmos_output"),
        )
        log(f"  checkpoint_path={setup_args.checkpoint_path}")

        engine_start = time.time()
        engine = Inference(args=setup_args)
        INFERENCE_ENGINE = engine

        log(f"Inference engine ready! engine_init={time.time() - engine_start:.1f}s, total={time.time() - load_start:.1f}s")
        log_memory_usage("after model load")
    except Exception as e:
        log(f"FATAL: Failed to create inference engine: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        raise

    return engine


# ── Media helpers ─────────────────────────────────────────────────────────────


def is_url(data: Any) -> bool:
    return isinstance(data, str) and (data.startswith("http://") or data.startswith("https://"))


def get_bytes(data: Any, timeout: int = 120) -> bytes:
    """Fetch bytes from a URL or decode from a base64 string."""
    if is_url(data):
        log(f"Downloading from URL: {data[:100]}...")
        t0 = time.time()
        resp = requests.get(data, timeout=timeout)
        resp.raise_for_status()
        log(f"Downloaded {len(resp.content)} bytes in {time.time() - t0:.1f}s")
        return resp.content
    return base64.b64decode(data.split("base64,")[-1])


def detect_media_type(data: bytes) -> str:
    """
    Detect media type from magic bytes.
    Returns one of: 'mp4', 'png', 'jpg', 'webp', or 'unknown'.
    """
    if data[:4] == b"\x89PNG":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    # MP4/MOV: 'ftyp' box at offset 4
    if len(data) >= 12 and data[4:8] in (b"ftyp", b"moov", b"mdat"):
        return "mp4"
    return "unknown"


def save_input_to_file(data: Any, temp_dir: str) -> tuple[str, str]:
    """
    Decode base64/URL data, detect type, save to temp file.
    Returns (file_path, media_type) where media_type is 'mp4'/'png'/'jpg'/'webp'.
    """
    raw = get_bytes(data)
    media_type = detect_media_type(raw)

    ext_map = {"mp4": ".mp4", "png": ".png", "jpg": ".jpg", "webp": ".webp", "unknown": ".bin"}
    ext = ext_map.get(media_type, ".bin")
    path = os.path.join(temp_dir, f"input{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    log(f"Saved input ({media_type}, {len(raw)} bytes) → {path}")
    return path, media_type


def infer_inference_type(media_type: str, explicit: str | None) -> str:
    """
    Determine inference_type string from media type and optional explicit override.
    Returns one of: 'text2world', 'image2world', 'video2world'.
    """
    if explicit is not None:
        return explicit
    if media_type in ("png", "jpg", "jpeg", "webp"):
        return "image2world"
    if media_type == "mp4":
        return "video2world"
    # fallback
    return "video2world"


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference_request(request_data: dict) -> Optional[str]:
    """Run a single inference request on the pre-loaded engine (all ranks call this)."""
    global INFERENCE_ENGINE

    if INFERENCE_ENGINE is None:
        raise RuntimeError("Inference engine not loaded!")

    kwargs = {
        "name": request_data.get("name", str(uuid.uuid4())[:8]),
        "prompt": request_data.get("prompt", "A video"),
        "inference_type": request_data.get("inference_type", "video2world"),
        "resolution": str(request_data.get("resolution", "none")),
        "num_output_frames": int(request_data.get("num_output_frames", 77)),
        "num_steps": int(request_data.get("num_steps", 35)),
        "seed": int(request_data.get("seed", 0)),
        "guidance": int(request_data.get("guidance", 7)),
        "enable_autoregressive": bool(request_data.get("enable_autoregressive", False)),
        "chunk_size": int(request_data.get("chunk_size", 77)),
        "chunk_overlap": int(request_data.get("chunk_overlap", 1)),
    }

    # input_path only for image2world / video2world
    if request_data.get("input_path"):
        kwargs["input_path"] = request_data["input_path"]

    # negative_prompt: only set if explicitly provided (model default otherwise)
    if request_data.get("negative_prompt"):
        kwargs["negative_prompt"] = request_data["negative_prompt"]

    sample = InferenceArguments(**kwargs)
    output_dir = Path(request_data["output_dir"])
    output_paths = INFERENCE_ENGINE.generate([sample], output_dir)

    return output_paths[0] if output_paths else None


# ── Multi-GPU coordination (NCCL broadcast) ───────────────────────────────────


def worker_loop():
    """Non-rank-0 processes: wait for broadcast, run inference, repeat."""
    rank = get_rank()
    log(f"Entering worker loop...")
    req_num = 0

    while True:
        signal = torch.zeros(1, dtype=torch.int32, device="cuda")
        dist.broadcast(signal, src=0)

        if signal.item() == 0:
            log(f"Shutdown signal received")
            break
        elif signal.item() == 1:
            req_num += 1
            data_len = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.broadcast(data_len, src=0)
            data_buf = torch.zeros(int(data_len.item()), dtype=torch.uint8, device="cuda")
            dist.broadcast(data_buf, src=0)

            request_data = json.loads(bytes(data_buf.cpu().numpy()).decode("utf-8"))
            log(f"Worker request #{req_num}, data_len={data_len.item()} bytes")
            try:
                run_inference_request(request_data)
                log(f"Worker request #{req_num} completed")
            except Exception as e:
                log(f"Worker request #{req_num} FAILED: {e}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")


def broadcast_inference(request_data: dict):
    """Rank 0: broadcast inference request dict to all worker ranks."""
    signal = torch.ones(1, dtype=torch.int32, device="cuda")
    dist.broadcast(signal, src=0)

    data_bytes = json.dumps(request_data).encode("utf-8")
    data_len = torch.tensor([len(data_bytes)], dtype=torch.int64, device="cuda")
    dist.broadcast(data_len, src=0)

    data_buf = torch.tensor(list(data_bytes), dtype=torch.uint8, device="cuda")
    dist.broadcast(data_buf, src=0)
    log(f"Broadcast {len(data_bytes)} bytes to {get_world_size()} workers")


def broadcast_shutdown():
    """Rank 0: send shutdown signal to all worker ranks."""
    signal = torch.zeros(1, dtype=torch.int32, device="cuda")
    dist.broadcast(signal, src=0)


# ── FastAPI application ───────────────────────────────────────────────────────


def create_fastapi_app():
    """Build and return the FastAPI app (rank 0 only)."""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    app = FastAPI(title="Cosmos Predict2.5 Persistent API", version="1.0.0")

    @app.post("/v1/infer")
    async def infer(request: Any = Body(...)):
        global REQUEST_COUNT
        REQUEST_COUNT += 1
        req_id = REQUEST_COUNT

        # Benchmarking clients may double-serialize the body
        if isinstance(request, (str, bytes)):
            request = json.loads(request)

        log(f"[Req #{req_id}] Received, keys: {list(request.keys())}")

        # Accept NIM API field names: "image" or "video" (preferred)
        # Also accept legacy field names for backwards compat: "input", "b64_input", "input_video"
        input_data = (
            request.get("image")
            or request.get("video")
            or request.get("input")
            or request.get("b64_input")
            or request.get("input_video")
        )

        # Determine inference type from NIM API convention:
        #   "image" field present → image2world
        #   "video" field present → video2world
        #   neither → text2world
        # Also accept explicit "inference_type" field for backwards compat
        inference_type_explicit = request.get("inference_type")
        if inference_type_explicit is None:
            if request.get("image") is not None:
                inference_type_explicit = "image2world"
            elif request.get("video") is not None:
                inference_type_explicit = "video2world"

        if input_data is None and inference_type_explicit not in (None, "text2world"):
            log(f"[Req #{req_id}] Missing visual input for inference_type={inference_type_explicit}", level="ERROR")
            raise HTTPException(400, f"Visual input required for inference_type={inference_type_explicit}")

        if input_data is None:
            inference_type_str = "text2world"
        else:
            inference_type_str = None  # resolved below after save

        temp_dir = tempfile.mkdtemp(dir="/tmp/cosmos_outputs")

        try:
            seed = request.get("seed")
            if seed is None:
                seed = random.randint(1, 2**16)

            output_dir = os.path.join(temp_dir, "output")

            # Map NIM API resolution keys to "H,W" format (matching NIM workflow)
            resolution = str(request.get("resolution", "720"))
            resolution_map = {
                "480": "432,768",   "480_1_1": "480,480",   "480_4_3": "640,480",
                "480_3_4": "480,640",   "480_9_16": "768,432",
                "720": "none",      "720_1_1": "960,960",   "720_4_3": "960,704",
                "720_3_4": "704,960",   "720_9_16": "1280,704",
                "1080": "1056,1920", "1080_1_1": "1024,1024", "1080_4_3": "1440,1056",
                "1080_3_4": "1056,1440", "1080_9_16": "1920,1056",
            }
            internal_resolution = resolution_map.get(resolution, resolution)

            # Accept NIM field names (guidance_scale, steps) with fallback to legacy names
            guidance = request.get("guidance_scale", request.get("guidance", 7))
            num_steps = request.get("steps", request.get("num_steps", 35))

            request_data: dict[str, Any] = {
                "name": str(uuid.uuid4())[:8],
                "output_dir": output_dir,
                "prompt": request.get("prompt", "A video"),
                "resolution": internal_resolution,
                "num_output_frames": int(request.get("num_output_frames", 77)),
                "num_steps": int(num_steps),
                "seed": seed,
                "guidance": float(guidance),
                "enable_autoregressive": request.get("enable_autoregressive", False),
                "chunk_size": request.get("chunk_size", 77),
                "chunk_overlap": request.get("chunk_overlap", 1),
            }

            if request.get("negative_prompt"):
                request_data["negative_prompt"] = request["negative_prompt"]

            if input_data is not None:
                input_path, media_type = save_input_to_file(input_data, temp_dir)
                inference_type_str = infer_inference_type(media_type, inference_type_explicit)
                request_data["input_path"] = input_path
                log(f"[Req #{req_id}] Input: media_type={media_type}, inference_type={inference_type_str}")
            else:
                inference_type_str = "text2world"

            request_data["inference_type"] = inference_type_str

            log(f"[Req #{req_id}] Inference params:")
            log(f"  model={model_name}, inference_type={inference_type_str}")
            log(f"  guidance_scale={request_data['guidance']}, steps={request_data['num_steps']}")
            log(f"  resolution={resolution} (internal={internal_resolution}), seed={seed}")
            log(f"  num_output_frames={request_data['num_output_frames']}, GPUs={get_world_size()}")

            t0 = time.time()

            with ENGINE_LOCK:
                if get_world_size() > 1:
                    broadcast_inference(request_data)
                output_path = run_inference_request(request_data)

            elapsed = time.time() - t0

            if not output_path or not os.path.exists(output_path):
                log(f"[Req #{req_id}] FAILED: no output after {elapsed:.2f}s", level="ERROR")
                return JSONResponse({"b64_video": None, "seed": seed, "error": "No video generated"})

            output_size = os.path.getsize(output_path)
            b64_video = base64.b64encode(open(output_path, "rb").read()).decode()

            log(f"[Req #{req_id}] SUCCESS: {elapsed:.2f}s, output={output_size} bytes, b64_len={len(b64_video)}")
            log_memory_usage(f"[Req #{req_id}] after inference")

            return JSONResponse({"b64_video": b64_video, "seed": seed})

        except HTTPException:
            raise
        except Exception as e:
            log(f"[Req #{req_id}] ERROR: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            raise HTTPException(500, str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.get("/v1/health/live")
    async def health_live():
        return {"status": "live"}

    @app.get("/v1/health/ready")
    async def health_ready():
        if INFERENCE_ENGINE is not None:
            return {"status": "ready"}
        raise HTTPException(503, "Model not loaded")

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": model_name,
            "model_loaded": INFERENCE_ENGINE is not None,
            "uptime_seconds": round(time.time() - STARTUP_TIME),
            "requests_served": REQUEST_COUNT,
            "world_size": get_world_size(),
        }

    @app.get("/")
    async def root():
        return {
            "message": "Cosmos Predict2.5 PERSISTENT API",
            "version": "1.0.0",
            "model": model_name,
            "model_loaded": INFERENCE_ENGINE is not None,
            "world_size": get_world_size(),
            "uptime_seconds": round(time.time() - STARTUP_TIME),
            "requests_served": REQUEST_COUNT,
        }

    return app


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    global STARTUP_TIME
    STARTUP_TIME = time.time()

    log("=" * 60, rank=0)
    log("Cosmos Predict2.5 PERSISTENT API — Starting", rank=0)
    log("=" * 60, rank=0)
    log(f"Python: {sys.version}", rank=0)
    log(f"PyTorch: {torch.__version__}", rank=0)
    log(f"CUDA available: {torch.cuda.is_available()}", rank=0)
    log(f"MODEL_NAME: {DEFAULT_MODEL}", rank=0)

    apply_nim_env_settings()

    # Keep NCCL alive between API calls (default 10 min would kill idle workers)
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "86400")

    if "RANK" in os.environ:
        log(
            f"Distributed mode: RANK={os.environ.get('RANK')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
            rank=0,
        )
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
        torch.cuda.set_device(get_rank())
    else:
        log("Single-process mode (no RANK env var)", rank=0)

    rank = get_rank()
    world_size = get_world_size()
    log(f"Process initialized: rank={rank}, world_size={world_size}, cuda_device={torch.cuda.current_device()}")

    if rank == 0:
        log_gpu_info()

    os.makedirs("/tmp/cosmos_outputs", exist_ok=True)

    log(f"Loading inference engine (model={DEFAULT_MODEL})...")
    load_inference_engine(model_name=DEFAULT_MODEL)

    if dist.is_initialized():
        log("Waiting at barrier for all ranks...")
        dist.barrier()
        log("All ranks synchronized")

    startup_elapsed = time.time() - STARTUP_TIME

    if rank == 0:
        import uvicorn

        app = create_fastapi_app()
        port = int(os.environ.get("API_PORT", "8000"))

        log("=" * 60)
        log("SERVER READY")
        log("=" * 60)
        log(f"  Model: {DEFAULT_MODEL}")
        log(f"  GPUs:  {world_size}")
        log(f"  Port:  {port}")
        log(f"  Inference types: text2world, image2world, video2world")
        log(f"  Defaults: guidance_scale=7.0, resolution=720 (internal=none), steps=35, num_output_frames=77")
        log(f"  NIM-compatible API: accepts 'image'/'video' fields, 'guidance_scale', 'steps', 'resolution'")
        log(f"  Also accepts legacy fields: 'input', 'guidance', 'num_steps' for backwards compat")
        log(f"  Startup time: {startup_elapsed:.1f}s")
        log("=" * 60)

        try:
            uvicorn.run(app, host="0.0.0.0", port=port)
        except Exception as e:
            log(f"FATAL: Uvicorn crashed: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
        finally:
            log("Shutting down...")
            if dist.is_initialized() and world_size > 1:
                broadcast_shutdown()
    else:
        worker_loop()

    if dist.is_initialized():
        dist.destroy_process_group()

    log("Process exited cleanly")


if __name__ == "__main__":
    main()
