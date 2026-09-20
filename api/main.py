"""Backend FastAPI.

Separa explicitamente as tres etapas cobradas pelo criterio de execucao tecnica:
    captura + processamento -> vision.Pipeline (thread propria)
    apresentacao            -> /video_feed (MJPEG) e /api/state (JSON)
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from vision.pipeline import Pipeline

app = FastAPI(
    title="Vaga Fantasma API",
    description="Ocupacao real de assentos em salas de estudo por visao computacional.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuracao por variavel de ambiente para que `run.py` e o uvicorn --reload
# compartilhem os mesmos parametros.
pipeline = Pipeline(
    source=os.getenv("VF_SOURCE", "0"),
    rois_path=os.getenv("VF_ROIS", "config/rois.json"),
    homography_path=os.getenv("VF_HOMOGRAPHY", "config/homography.json"),
    model_path=os.getenv("VF_MODEL", "yolov8n.pt"),
    conf=float(os.getenv("VF_CONF", "0.25")),
    iou_threshold=float(os.getenv("VF_IOU", "0.25")),
    ghost_after_s=float(os.getenv("VF_GHOST_AFTER", "20")),
    imgsz=int(os.getenv("VF_IMGSZ", "512")),
    detect_every_s=float(os.getenv("VF_DETECT_EVERY", "0.2")),
    dynamic=os.getenv("VF_DYNAMIC", "0") == "1",
)


@app.on_event("startup")
def _startup():
    pipeline.start()


@app.on_event("shutdown")
def _shutdown():
    pipeline.stop()


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse("web/index.html")


@app.get("/health", summary="Health check")
def health():
    return {"status": "healthy", "service": "vaga-fantasma", "fps": pipeline.get_state()["fps"]}


@app.get("/api/state", summary="Estado atual de cada assento")
def state():
    """Consumido pelo dashboard a cada ~500 ms."""
    return pipeline.get_state()


def _mjpeg_frames():
    import time

    while True:
        frame = pipeline.get_jpeg()
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)  # ~20 fps de envio, para nao disputar CPU com a inferencia


@app.get("/video_feed", summary="Stream MJPEG do video anotado")
def video_feed():
    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
