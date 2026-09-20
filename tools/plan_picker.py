#!/usr/bin/env python3
"""Calibra a homografia: marca no chao os 4 cantos de um retangulo real.

Uso:
    python tools/plan_picker.py --source 0
    python tools/plan_picker.py --source http://192.168.0.X:8080/live --largura 6 --profundidade 4

Clique os 4 cantos NESTA ORDEM (e importante):
    1. frente-esquerda   (canto proximo da camera, a esquerda)
    2. frente-direita
    3. fundo-direita
    4. fundo-esquerda

"Frente" e o lado mais proximo da camera. Escolha cantos que voce saiba
formarem um retangulo no mundo real: os pes de uma mesa, o rodape da
sala, as juntas do piso. Quanto maior a area coberta, melhor a precisao.

Controles:
    clique  marca um canto      u  desfaz      s  salva      q/ESC  sai
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.capture import VideoSource  # noqa: E402
from vision.floorplan import FloorPlan  # noqa: E402

ROTULOS = ["1 frente-esquerda", "2 frente-direita", "3 fundo-direita", "4 fundo-esquerda"]
pontos = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
        pontos.append((x, y))


def desenha_previa(view, pts, w, h):
    """Mostra a planta resultante num canto, para conferir antes de salvar."""
    plano = FloorPlan([(x / w, y / h) for x, y in pts], (1, 1))
    pw, ph = 200, 140
    ox, oy = w - pw - 14, 14

    cv2.rectangle(view, (ox, oy), (ox + pw, oy + ph), (30, 30, 30), -1)
    cv2.rectangle(view, (ox, oy), (ox + pw, oy + ph), (200, 200, 200), 1)
    cv2.putText(view, "planta", (ox + 6, oy + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Grade projetada: se a calibracao estiver boa, sai quadriculada e regular.
    for i in range(5):
        t = i / 4
        cv2.line(view, (int(ox + t * pw), oy), (int(ox + t * pw), oy + ph), (70, 70, 70), 1)
        cv2.line(view, (ox, int(oy + t * ph)), (ox + pw, int(oy + t * ph)), (70, 70, 70), 1)
    return plano


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--out", default="config/homography.json")
    parser.add_argument("--largura", type=float, default=6.0,
                        help="largura real da area marcada, em metros")
    parser.add_argument("--profundidade", type=float, default=4.0,
                        help="profundidade real da area marcada, em metros")
    args = parser.parse_args()

    win = "Plan Picker - clique os 4 cantos do chao | u=desfaz s=salva q=sai"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    with VideoSource(args.source) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            view = frame.copy()

            for i, (x, y) in enumerate(pontos):
                cv2.circle(view, (x, y), 7, (0, 180, 255), -1)
                cv2.putText(view, str(i + 1), (x + 11, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
            if len(pontos) > 1:
                traco = np.int32(pontos).reshape(-1, 1, 2)
                cv2.polylines(view, [traco], len(pontos) == 4, (0, 180, 255), 2)

            if len(pontos) < 4:
                msg = f"Clique: {ROTULOS[len(pontos)]}"
            else:
                msg = "4 cantos marcados - 's' para salvar"
                desenha_previa(view, pontos, w, h)

            cv2.rectangle(view, (0, h - 34), (w, h), (0, 0, 0), -1)
            cv2.putText(view, msg, (10, h - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and pontos:
                pontos.pop()
            if key == ord("s"):
                if len(pontos) != 4:
                    print("Marque os 4 cantos antes de salvar.")
                    continue
                payload = {
                    "image_points": [[x / w, y / h] for x, y in pontos],
                    "plan_size_m": [args.largura, args.profundidade],
                    "reference_resolution": [w, h],
                }
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                print(f"Homografia salva em {args.out} "
                      f"({args.largura}m x {args.profundidade}m)")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
