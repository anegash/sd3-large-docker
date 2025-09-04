# Generic Dummy Docker Container Prompt

## Prompt to Generate Empty Development Container

"Create a minimal dummy Docker container setup for development purposes with the following requirements:

### Base Requirements:
1. Use an appropriate base image (Ubuntu 22.04, Python 3.10, or similar)
2. Keep container alive indefinitely for SSH access
3. Install only essential tools: git, curl, wget, nano/vim
4. Create a simple keep-alive script that prevents container shutdown
5. Set up basic directory structure (/app, /workspace, /data)
6. Include environment variable setup for common paths

### The container should:
- Start and stay running forever (no automatic service startup)
- Allow SSH access for manual operations
- Support git operations for pulling code updates
- Have Python/Node/etc runtime based on project needs
- Display helpful startup message with available commands
- Log heartbeat messages to prove activity to the host platform

### Files to create:
1. `Dockerfile.dummy` - Minimal Dockerfile
2. `dummy_keepalive.sh` - Script that keeps container running
3. Build and deployment instructions

### Key features:
- No application-specific code or services auto-start
- Manual control over all processes
- Suitable for iterative development without Docker rebuilds
- Platform-agnostic (works on RunPod, AWS, local, etc.)
- Prevents timeout/shutdown on cloud platforms

The goal is a reusable template for any project where I need a persistent development container with manual control over services."

---

## Example Usage for Future Projects:

```bash
# 1. Generate the dummy container files using the prompt above

# 2. Build the container
docker build -f Dockerfile.dummy -t myproject:dev-dummy .

# 3. Deploy to cloud platform (RunPod, AWS, etc.)
docker push myproject:dev-dummy

# 4. SSH into running container
ssh user@container-host

# 5. Pull your actual code
git clone https://github.com/username/project.git
cd project

# 6. Start services manually as needed
python app.py
npm start
./run.sh
```

## Template Dockerfile.dummy:

```dockerfile
FROM ubuntu:22.04

# Install basics
RUN apt-get update && apt-get install -y \
    git curl wget nano vim \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /app /workspace /data

WORKDIR /app

# Copy keep-alive script
COPY dummy_keepalive.sh /app/
RUN chmod +x /app/dummy_keepalive.sh

# Expose common ports (optional)
EXPOSE 3000 5000 8000 8080

# Run keep-alive script
CMD ["/app/dummy_keepalive.sh"]
```

## Template dummy_keepalive.sh:

```bash
#!/bin/bash

echo "================================================"
echo "🔧 DEVELOPMENT CONTAINER READY"
echo "================================================"
echo "Container will stay alive for manual development"
echo ""
echo "Common commands:"
echo "  git clone <repo>     - Clone your project"
echo "  python3 script.py    - Run Python scripts"
echo "  ps aux               - Check running processes"
echo "  kill -9 <pid>        - Stop a process"
echo ""
echo "Container started at: $(date)"
echo "================================================"

# Keep container alive with periodic heartbeat
while true; do
    echo "[$(date)] Container heartbeat - keeping alive for development" >> /tmp/heartbeat.log
    sleep 60
done
```

This prompt and templates can be reused for any project requiring a persistent development container!