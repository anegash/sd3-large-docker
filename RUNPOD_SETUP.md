# RunPod Quick Setup

## One-time setup after pod restart:

```bash
cd /workspace/sd3-large-docker && ./init_runpod.sh && source ~/.bashrc
```

## Then start the server:

```bash
sd3-start
# or
/workspace/start_sd3.sh
```

## If you get "poetry not found" error:

```bash
export PATH="/root/.local/bin:$PATH" && /workspace/start_sd3.sh
```

## Complete setup from scratch:

```bash
# If needed, clone first:
# git clone -b feature/lora-training-runpod https://github.com/anegash/sd3-large-docker.git /workspace/sd3-large-docker

cd /workspace/sd3-large-docker
./init_runpod.sh
source ~/.bashrc
sd3-start
```