# Cosmos Predict2.5 — Persistent Benchmark API

`nim_api_persistent.py` is a FastAPI-based inference server designed for performance
benchmarking of Cosmos Predict2.5. It is compatible with the `benchmarking-speed`
client used in JET CI and exposes a `/v1/infer` endpoint that mirrors the NIM API schema.

## Why a persistent server?

The standard CLI (`cosmos_predict2/inference.py`) spawns a fresh process for every
request, incurring heavy model-load overhead each time. This server loads the model
once at startup and handles subsequent requests in-memory.

## Architecture

- **Rank 0** runs the FastAPI/uvicorn HTTP server and handles request parsing.
- **Ranks 1…N** run a worker loop, receiving inference jobs via NCCL broadcast.
- All ranks participate in each forward pass (context parallelism).

## Model selection

The model is selected at startup via the `MODEL_NAME` environment variable.
Default: `2B/post-trained`.

| `MODEL_NAME` | Description |
|---|---|
| `2B/pre-trained` | 2B parameters, pre-trained |
| `2B/post-trained` | 2B parameters, post-trained **(default)** |
| `2B/distilled` | 2B parameters, distilled (fewer steps needed) |
| `14B/pre-trained` | 14B parameters, pre-trained |
| `14B/post-trained` | 14B parameters, post-trained |

## Building the container

`nim_api_persistent.py`, `fastapi`, and `uvicorn` are all included in the standard
nightly image. No separate build step is required.

```bash
docker build -f docker/nightly.Dockerfile -t cosmos-predict2.5:nightly .
```

To use a different PyTorch base image pass `--build-arg BASE_IMAGE=...`.

## Running

Model weights must be available in the HF cache inside the container. Either bake them
in during the build or mount them at runtime.

### Single GPU
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  -v /path/to/hf_cache:/root/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 \
  -p 8000:8000 \
  cosmos-predict2.5:nightly \
  python3 /workspace/nim_api_persistent.py
```

### Multi-GPU (e.g. 4 GPUs)
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  -v /path/to/hf_cache:/root/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 \
  -e MASTER_ADDR=localhost \
  -e MASTER_PORT=29500 \
  -p 8000:8000 \
  cosmos-predict2.5:nightly \
  torchrun --nproc_per_node=4 /workspace/nim_api_persistent.py
```

### Different model
```bash
docker run ... -e MODEL_NAME=14B/post-trained ... torchrun --nproc_per_node=8 ...
```

## Health endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/health/live` | Returns 200 immediately |
| `GET /v1/health/ready` | Returns 200 once model is loaded, 503 otherwise |
| `GET /health` | Returns model name, uptime, request count, GPU count |

## Request schema (`POST /v1/infer`)

```json
{
  "prompt": "A robot navigating a warehouse",
  "input": "<base64-encoded image/video or https:// URL>",
  "inference_type": "video2world",
  "resolution": "none",
  "num_output_frames": 77,
  "num_steps": 35,
  "seed": 42,
  "guidance": 7,
  "negative_prompt": null,
  "enable_autoregressive": false,
  "chunk_size": 77,
  "chunk_overlap": 1
}
```

### `inference_type` values

| Value | Input required | Description |
|---|---|---|
| `text2world` | No | Generate video from text prompt only |
| `image2world` | Yes (image) | Generate video conditioned on a single image |
| `video2world` | Yes (image or video) | Generate video conditioned on image/video **(default when input provided)** |

If `inference_type` is omitted:
- No `input` → `text2world`
- `input` provided → auto-detected from file type (PNG/JPG → `image2world`, MP4 → `video2world`)

`input` accepts a base64-encoded PNG, JPEG, WEBP, or MP4, or a public HTTPS URL.

## Response

```json
{
  "b64_video": "<base64-encoded output mp4>",
  "seed": 42
}
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `2B/post-trained` | Model to load at startup |
| `API_PORT` | `8000` | HTTP port to listen on |
| `HF_HUB_OFFLINE` | `1` | Use only locally cached model weights |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `86400` | NCCL heartbeat timeout (24h keeps workers alive between requests) |
