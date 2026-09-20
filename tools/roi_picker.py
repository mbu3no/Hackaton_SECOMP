#!/usr/bin/env python3
"""Calibrador de ROIs: marca os assentos com o mouse sobre a imagem da camera.

Uso:
    python tools/roi_picker.py --source 0
    python tools/roi_picker.py --source samples/demo.mp4 --out config/rois.json

Controles:
    arrastar o mouse  desenha a ROI de um assento
    u                 desfaz a ultima ROI
    s                 salva e sai
    q / ESC           sai sem salvar

As ROIs sao gravadas em coordenadas normalizadas [0..1], entao continuam
validas se a resolucao da camera mudar entre a calibracao e a demo.
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.capture import VideoSource  # noqa: E402

rois = []
drawing = False
start_pt = (0, 0)
current = None


def on_mouse(event, x, y, flags, param):
    global drawing, start_pt, current
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)
        current = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current = (start_pt[0], start_pt[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_pt[0], x), min(start_pt[1], y)
        x2, y2 = max(start_pt[0], x), max(start_pt[1], y)
        if x2 - x1 > 10 and y2 - y1 > 10:  # ignora clique acidental
            rois.append((x1, y1, x2, y2))
        current = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--out", default="config/rois.json")
    args = parser.parse_args()

    win = "ROI Picker - arraste para marcar assentos | u=desfaz s=salva q=sai"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    with VideoSource(args.source) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            view = frame.copy()

            for i, (x1, y1, x2, y2) in enumerate(rois):
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(view, f"A{i + 1}", (x1 + 4, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

            if current is not None:
                cv2.rectangle(view, current[:2], current[2:], (0, 180, 255), 2)

            cv2.putText(view, f"{len(rois)} assento(s)", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and rois:
                rois.pop()
            if key == ord("s"):
                payload = {
                    "reference_resolution": [w, h],
                    "seats": [
                        {"id": f"A{i + 1}", "roi": [x1 / w, y1 / h, x2 / w, y2 / h]}
                        for i, (x1, y1, x2, y2) in enumerate(rois)
                    ],
                }
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False)
                print(f"{len(rois)} assento(s) salvos em {args.out}")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
