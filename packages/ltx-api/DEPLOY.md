# LTX-2 API — Bare-Metal Deployment Guide

Deploy the FastAPI inference server on a cloud GPU instance without Docker.

---

## 1. Provisioning a GPU Instance

### Recommended Specs

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA A10G (24 GB) | NVIDIA A100 (80 GB) or H100 |
| VRAM | 24 GB (fp8-cast) | 48 GB+ (bf16) |
| CPU | 8 vCPUs | 16 vCPUs |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 200 GB SSD |
| OS | Ubuntu 24.04 LTS | Ubuntu 24.04 LTS |

### Supported Cloud Providers

| Provider | Instance Type | GPU | Approx. Cost |
|---|---|---|---|
| **AWS** | `g5.2xlarge` | A10G 24 GB | ~$1.00/hr |
| **AWS** | `p4d.24xlarge` | A100 40 GB × 8 | ~$32.00/hr |
| **GCP** | `g2-standard-8` | L4 24 GB | ~$0.70/hr |
| **GCP** | `a2-highgpu-1g` | A100 40 GB | ~$3.50/hr |
| **Azure** | `NC6s v3` | V100 16 GB | ~$3.00/hr |
| **Azure** | `ND96asr v4` | A100 80 GB | ~$15.00/hr |
| **Lambda Labs** | `gpu_1x_a10` | A10G 24 GB | ~$0.75/hr |
| **Vast.ai** | Various | RTX 4090 24 GB | ~$0.30/hr |
| **RunPod** | `SECURE` | A100 80 GB | ~$2.50/hr |

> **Note:** FP8 quantization (`LTX_QUANTIZATION=fp8-cast`) reduces VRAM usage by ~40%. Use it for 24 GB GPUs.

---

## 2. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and common build deps
sudo apt install -y python3.11 python3.11-dev python3-pip python3-venv \
    git curl wget build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg

# Verify Python
python3.11 --version
```

---

## 3. NVIDIA Drivers & CUDA

### Option A: Install via NVIDIA CUDA Toolkit (recommended)

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent --override

# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify
nvidia-smi
nvcc --version
```

### Option B: Install GPU drivers only (use cloud provider's CUDA image)

Most cloud GPU images come with drivers pre-installed. Verify:

```bash
nvidia-smi
```

You should see output showing your GPU model, driver version, and CUDA version (12.4+).

---

## 4. Application Setup

```bash
# Clone the repository
git clone https://github.com/buntercodes/LTX-2.3-API-Infer.git /home/ubuntu/ltx-api
cd /home/ubuntu/ltx-api

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create virtual environment and sync dependencies
uv sync --frozen

# Verify the installation
uv run python -c "from ltx_pipelines.distilled import DistilledPipeline; print('OK')"
```

---

## 5. Download Model Weights

All weights are hosted on [HuggingFace — Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3).

### Required Files for DistilledPipeline

| File | Size | Source |
|---|---|---|
| `ltx-2.3-22b-distilled.safetensors` | ~44 GB | HuggingFace |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | ~100 MB | HuggingFace |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | ~7 GB | Google |

```bash
# Create model directory
sudo mkdir -p /models
sudo chown -R $(whoami):$(whoami) /models

# Install HuggingFace CLI
pip install huggingface-hub

# Download distilled checkpoint
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled.safetensors \
    --local-dir /models/ltx-2.3 \
    --local-dir-use-symlinks False

# Download spatial upsampler
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --local-dir /models/ltx-2.3 \
    --local-dir-use-symlinks False

# Download Gemma text encoder (requires HF acceptance)
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir /models/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir-use-symlinks False
```

> **Note:** Gemma access requires accepting the license on HuggingFace. Log in first:
> ```bash
> huggingface-cli login
> ```

---

## 6. Configuration

Create the `.env` file with all required settings, then create the output directory.

### Step 1 — Create the output directory

```bash
mkdir -p /home/ubuntu/ltx-api/output
```

### Step 2 — Create the .env file

```bash
nano /home/ubuntu/ltx-api/.env
```

### Step 3 — Fill the .env file with the values below

Paste this content, then adjust any paths that differ on your instance:

```bash
# ---- REQUIRED: Model paths ----
# Point each path to the files you downloaded in step 5

LTX_DISTILLED_CHECKPOINT_PATH=/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors
LTX_SPATIAL_UPSAMPLER_PATH=/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
LTX_GEMMA_ROOT=/models/gemma-3-12b-it-qat-q4_0-unquantized

# ---- OPTIONAL: LoRAs ----
# Leave empty if you don't use LoRAs. Otherwise list comma-separated paths.
LTX_LORA_PATHS=
LTX_LORA_STRENGTHS=

# ---- OPTIONAL: Quantization ----
# Choices: "" (bf16, default), "fp8-cast" (24 GB GPUs), "fp8-scaled-mm" (Hopper GPUs)
LTX_QUANTIZATION=fp8-cast

# ---- Server ----
LTX_API_HOST=0.0.0.0
LTX_API_PORT=8000

# ---- Output storage ----
LTX_OUTPUT_DIR=/home/ubuntu/ltx-api/output
LTX_MAX_OUTPUT_AGE_HOURS=24
```

Save the file (`Ctrl+O`, then `Ctrl+X` in nano).

### Step 4 — Verify the .env file

```bash
cat /home/ubuntu/ltx-api/.env
```

Confirm every path points to an actual file or directory:

```bash
ls -lh "$LTX_DISTILLED_CHECKPOINT_PATH"
ls -lh "$LTX_SPATIAL_UPSAMPLER_PATH"
ls -d "$LTX_GEMMA_ROOT"
```

---

## 7. Running the Server

### Direct Start

```bash
cd /home/ubuntu/ltx-api
source .env
uv run python -m ltx_api
```

The server will:
1. Load the pipeline (can take 2–5 minutes depending on storage speed)
2. Print log output showing model loading progress
3. Start listening on `0.0.0.0:8000`

### Systemd Service (production)

```bash
sudo tee /etc/systemd/system/ltx-api.service << 'SERVICEEOF'
[Unit]
Description=LTX-2 Distilled Inference API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/ltx-api
EnvironmentFile=/home/ubuntu/ltx-api/.env
ExecStart=/home/ubuntu/ltx-api/.venv/bin/python -m ltx_api
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

# GPU isolation
RuntimeDirectory=ltx-api
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable ltx-api
sudo systemctl start ltx-api
sudo systemctl status ltx-api
```

### Viewing Logs

```bash
sudo journalctl -u ltx-api -f
```

---

## 8. Verifying the API

```bash
# Health check
curl -s http://localhost:8000/health | jq

# Expected output:
# {
#   "status": "ok",
#   "gpu_available": true,
#   "gpu_name": "NVIDIA A100 80GB PCIe",
#   "queue_length": 0,
#   "active_task": null
# }

# Submit a generation request
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset, slow camera pan",
    "seed": 42,
    "height": 512,
    "width": 768,
    "num_frames": 121,
    "frame_rate": 24
  }' | jq

# Expected output:
# {
#   "task_id": "a1b2c3d4e5f6g7h8",
#   "status": "queued",
#   ...
# }

# Poll task status (replace with your task_id)
curl -s http://localhost:8000/api/v1/tasks/a1b2c3d4e5f6g7h8 | jq

# Download completed video
curl -s http://localhost:8000/api/v1/tasks/a1b2c3d4e5f6g7h8/download \
  -o output.mp4
```

---

## 9. Memory Optimization

### FP8 Quantization (24 GB GPUs)

Set `LTX_QUANTIZATION=fp8-cast` to use FP8 casting. This downcasts transformer linear weights to FP8 during loading and upcasts on the fly. Reduces VRAM by ~40%.

For Hopper GPUs (H100, H200), set `LTX_QUANTIZATION=fp8-scaled-mm` for additional speedup via TensorRT-LLM. This requires installing `tensorrt-llm`:

```bash
uv pip install tensorrt-llm==1.0.0 onnx openmpi
```

### Gradient Checkpointing

Not applicable to inference, but ensure memory cleanup is working:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Disable Memory Cleanup (if you have sufficient VRAM)

The distilled pipeline automatically cleans GPU memory between stages. If you have 80 GB+ VRAM, this adds unnecessary overhead. To disable it, patch `pipeline.py`:

```python
# In packages/ltx-api/src/ltx_api/pipeline.py, before calling pipeline():
# torch.cuda.empty_cache() calls are handled inside DistilledPipeline
```

---

## 10. Production Hardening

### Reverse Proxy with Nginx

```bash
sudo apt install -y nginx

sudo tee /etc/nginx/sites-available/ltx-api << 'NGINXEOF'
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 10m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
    }

    location /api/v1/tasks/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_read_timeout 600s;
    }
}
NGINXEOF

sudo ln -sf /etc/nginx/sites-available/ltx-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### SSL with Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Rate Limiting (nginx)

Add inside the `server` block:

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=5r/m;
location /api/v1/generate {
    limit_req zone=api burst=1 nodelay;
    proxy_pass http://127.0.0.1:8000;
}
```

### Firewall

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## 11. Client Usage Examples

### Python

```python
import requests

API_BASE = "http://your-instance:8000"

# Submit generation
resp = requests.post(
    f"{API_BASE}/api/v1/generate",
    json={
        "prompt": "Cinematic drone shot of a futuristic city at night",
        "seed": 42,
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "frame_rate": 24,
    },
)
task = resp.json()
task_id = task["task_id"]

# Poll until completed
import time
while True:
    resp = requests.get(f"{API_BASE}/api/v1/tasks/{task_id}")
    status = resp.json()
    print(status["status"], status.get("progress"))
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(5)

# Download result
resp = requests.get(f"{API_BASE}/api/v1/tasks/{task_id}/download")
with open("output.mp4", "wb") as f:
    f.write(resp.content)
```

### curl

```bash
# Submit with image conditioning
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat wearing a hat, walking on a beach",
    "images": [
      {"path": "https://example.com/cat.jpg", "frame_idx": 0, "strength": 0.8}
    ],
    "height": 512,
    "width": 768,
    "num_frames": 121,
    "frame_rate": 24
  }' | jq
```

### Image Conditioning

When providing image conditioning, the `path` field must be a local path accessible to the server. To use remote URLs, download them first on the server or mount a shared filesystem.

---

## 12. Monitoring

### GPU Metrics

```bash
# Watch GPU usage in real time
watch -n 1 nvidia-smi

# Detailed GPU metrics
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### Application Metrics

The `/health` endpoint provides:

```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA A100 80GB PCIe",
  "queue_length": 2,
  "active_task": "a1b2c3d4e5f6g7h8"
}
```

### Prometheus + Grafana (optional)

Install the [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator) for metrics export.

---

## 13. Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory.
```

**Solutions:**
1. Enable FP8: `LTX_QUANTIZATION=fp8-cast`
2. Reduce resolution: use `height: 384, width: 576`
3. Reduce frames: use `num_frames: 65` instead of 121
4. Upgrade to a GPU with more VRAM

### Model Loading Hangs

If the server hangs during startup:

```bash
# Check disk I/O — loading 44 GB checkpoint can be slow on network storage
iostat -x 5

# Try a faster disk (local NVMe instead of NFS)
cp /models/ltx-2.3-22b-distilled.safetensors /tmp/
LTX_DISTILLED_CHECKPOINT_PATH=/tmp/ltx-2.3-22b-distilled.safetensors
```

### Slow Generation

Generation time depends on GPU, resolution, and frame count. Approximate times (fp8-cast, 121 frames at 512×768):

| GPU | Time |
|---|---|
| A10G (24 GB) | ~90–120s |
| A100 (80 GB) | ~45–60s |
| H100 (80 GB) | ~25–35s |

### Generation Fails with "assert_resolution"

```
ValueError: Resolution (512x770) is not divisible by 64.
```

Ensure your `height` and `width` are multiples of 64 (e.g., 512, 576, 640, 704, 768) and `(num_frames - 1) % 8 == 0` (e.g., 65, 97, 121).

---

## 14. Updating

```bash
cd /home/ubuntu/ltx-api
git pull origin main
uv sync --frozen
sudo systemctl restart ltx-api
```
