FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# full: YOLO + PyTorch + MediaPipe, lite: MediaPipe only (much smaller image)
ARG EDITION=full

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO26N_HOST=0.0.0.0 \
    YOLO26N_PORT=8765 \
    YOLO26N_EDITION=${EDITION} \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./
RUN if [ "$EDITION" = "full" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra full; \
    elif [ "$EDITION" = "lite" ]; then \
        uv sync --frozen --no-dev --no-install-project; \
    else \
        echo "Unknown EDITION=$EDITION (use full or lite)" >&2; exit 1; \
    fi

COPY bin ./bin
COPY config ./config
COPY driver_files ./driver_files
COPY models ./models
COPY face_landmarker.task hand_landmarker.task ./
COPY yolo26n-pose.pt yolo26n.pt ./
RUN if [ "$EDITION" = "lite" ]; then rm -f yolo26n-pose.pt yolo26n.pt bin/yolo_detector.py; fi

EXPOSE 8765 9000/udp 9001/udp 39539/udp 39540/udp 39541/udp 39570/udp 4243/udp

CMD ["python", "bin/desktop_launcher.py", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]