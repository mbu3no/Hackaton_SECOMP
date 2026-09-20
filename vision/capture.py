"""Fonte de video plugavel: webcam, arquivo ou camera IP.

A fonte e trocavel por uma string, o que garante o plano B da demo ao vivo:
se a webcam falhar no pitch, basta apontar para um arquivo .mp4 e o resto
do pipeline continua identico.
"""

import sys
import cv2


class VideoSource:
    """Abre uma fonte de video e entrega frames BGR.

    source:
        "0", "1", ...        -> indice de webcam (inclui celular via Iriun/DroidCam)
        "samples/demo.mp4"   -> arquivo de video
        "rtsp://..."         -> camera IP
    """

    def __init__(self, source: str = "0", width: int = 1280, height: int = 720, loop: bool = True):
        self.source = source
        self.loop = loop
        self._is_file = not source.isdigit() and not source.startswith("rtsp://")

        if source.isdigit():
            # CAP_DSHOW evita o delay de ~3s na abertura da webcam no Windows
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(source), backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source!r}")

    def read(self):
        """Retorna o proximo frame BGR, ou None se a fonte acabou."""
        ok, frame = self.cap.read()

        # Arquivo de video chegou ao fim: reinicia para a demo rodar em loop
        if not ok and self._is_file and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        return frame if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
