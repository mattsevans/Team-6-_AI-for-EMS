# -------------------------------
# AI Subsystem — Dockerfile
# Runtime: Python 3.13 (multi-arch, incl. arm64 for Raspberry Pi)
# -------------------------------
FROM python:3.13-slim

# Environment: no .pyc, unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-liberation \
  && rm -rf /var/lib/apt/lists/*

  # Make chromium discoverable under common names
RUN ln -sf /usr/bin/chromium /usr/bin/google-chrome && \
    ln -sf /usr/bin/chromium /usr/bin/chromium-browser


# If any package needs compiling on 3.13, keep this; otherwise you can comment it out to slim the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer-cached)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    # Try to use wheels only; if a package lacks a 3.13 wheel and you need it,
    # drop the --only-binary flag or fall back to Python 3.11 later.
    pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy source code
COPY . .

# Ensure runtime folders exist if your code expects them
RUN mkdir -p /app/reports

# Create non-root user and switch to it
RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

# Default entrypoint: your mailbox watcher loop
CMD ["python", "edge_loop_tie.py"]