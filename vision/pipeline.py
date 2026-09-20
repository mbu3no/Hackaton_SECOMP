"""Orquestracao: captura -> processamento -> saida.

Roda em uma thread propria para que a API possa servir o video anotado e o
estado em JSON sem bloquear.
"""

import threading
import time
from typing import Dict, List, Optional

import cv2

from .capture import VideoSource
from .detector import Detector
from .floorplan import FloorPlan
from .seats import Seat, SeatState, load_seats

# Cores em BGR
COLORS = {
    SeatState.LIVRE: (0, 200, 0),
    SeatState.OCUPADO: (60, 60, 220),
    SeatState.FANTASMA: (0, 180, 255),
}


class Pipeline:
    def __init__(
        self,
        source: str = "0",
        rois_path: str = "config/rois.json",
        homography_path: str = "config/homography.json",
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        iou_threshold: float = 0.25,
        ghost_after_s: float = 20.0,
    ):
        self.source_str = source
        self.iou_threshold = iou_threshold
        self.ghost_after_s = ghost_after_s

        self.detector = Detector(model_path=model_path, conf=conf)
        self.seats: List[Seat] = load_seats(rois_path)

        # Opcional: sem calibracao de homografia o sistema roda igual,
        # apenas sem o painel de planta baixa.
        self.plan: Optional[FloorPlan] = FloorPlan.load(homography_path)
        self._seat_plan_xy = self._project_seats()
        self._people_plan_xy: List[List[float]] = []

        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0

    # ------------------------------------------------------------------ ciclo

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self):
        with VideoSource(self.source_str) as cam:
            last = time.time()
            while self._running:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                detections = self.detector.detect(frame)
                for seat in self.seats:
                    seat.observe(detections, self.iou_threshold, self.ghost_after_s)

                if self.plan is not None:
                    self._people_plan_xy = [
                        list(self.plan.project_box((d.x1, d.y1, d.x2, d.y2)))
                        for d in detections if d.is_person
                    ]

                now = time.time()
                self._fps = 1.0 / max(now - last, 1e-6)
                last = now

                annotated = self._render(frame)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._frame = buf.tobytes()

    # ------------------------------------------------------------------ saida

    def _render(self, frame):
        """Desenha as ROIs coloridas pelo estado atual de cada assento."""
        h, w = frame.shape[:2]
        out = frame.copy()

        for seat in self.seats:
            x1, y1, x2, y2 = seat.roi
            p1 = (int(x1 * w), int(y1 * h))
            p2 = (int(x2 * w), int(y2 * h))
            color = COLORS[seat.state]

            cv2.rectangle(out, p1, p2, color, 2)

            caption = f"{seat.seat_id}: {seat.state.value}"
            if seat.state is SeatState.FANTASMA:
                caption += f" ({int(seat.abandoned_for)}s)"

            cv2.rectangle(out, (p1[0], p1[1] - 20), (p1[0] + 9 * len(caption), p1[1]), color, -1)
            cv2.putText(out, caption, (p1[0] + 3, p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(out, f"{self._fps:4.1f} FPS", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def _project_seats(self) -> Dict[str, List[float]]:
        """Posicao de cada assento na planta, derivada da homografia.

        Os assentos nao sao posicionados a mao no mapa: a posicao vem da
        projecao da base da ROI no plano do chao.
        """
        if self.plan is None:
            return {}
        return {
            seat.seat_id: list(self.plan.project_box(seat.roi))
            for seat in self.seats
        }

    def get_state(self) -> Dict:
        counts = {s.value: 0 for s in SeatState}
        for seat in self.seats:
            counts[seat.state.value] += 1

        seats = []
        for seat in self.seats:
            item = seat.to_dict()
            item["plan_xy"] = self._seat_plan_xy.get(seat.seat_id)
            seats.append(item)

        return {
            "fps": round(self._fps, 1),
            "total": len(self.seats),
            "counts": counts,
            "ghost_after_s": self.ghost_after_s,
            "seats": seats,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "people_plan_xy": self._people_plan_xy,
        }
