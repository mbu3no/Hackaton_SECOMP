#!/usr/bin/env python3
"""Ponto de entrada.

Exemplos:
    python run.py                                  # webcam padrao
    python run.py --source 1                       # celular via Iriun/DroidCam
    python run.py --source samples/demo.mp4        # plano B da demo ao vivo
    python run.py --ghost-after 10                 # tolerancia menor para o pitch
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Vaga Fantasma - SECOMP 2026")
    parser.add_argument("--source", default="0",
                        help="indice da webcam, caminho de video ou url rtsp://")
    parser.add_argument("--rois", default="config/rois.json",
                        help="arquivo de ROIs gerado por tools/roi_picker.py")
    parser.add_argument("--homography", default="config/homography.json",
                        help="calibracao da planta baixa (tools/plan_picker.py)")
    parser.add_argument("--model", default="yolov8n.pt", help="peso YOLO")
    parser.add_argument("--conf", type=float, default=0.35, help="confianca minima")
    parser.add_argument("--iou", type=float, default=0.25,
                        help="sobreposicao minima entre deteccao e ROI")
    parser.add_argument("--ghost-after", type=float, default=20.0,
                        help="segundos sem pessoa ate o assento virar fantasma")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="lado da imagem na entrada da rede (menor = mais rapido)")
    parser.add_argument("--detect-every", type=float, default=0.2,
                        help="intervalo minimo entre inferencias, em segundos")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["VF_SOURCE"] = args.source
    os.environ["VF_ROIS"] = args.rois
    os.environ["VF_HOMOGRAPHY"] = args.homography
    os.environ["VF_MODEL"] = args.model
    os.environ["VF_CONF"] = str(args.conf)
    os.environ["VF_IOU"] = str(args.iou)
    os.environ["VF_GHOST_AFTER"] = str(args.ghost_after)
    os.environ["VF_IMGSZ"] = str(args.imgsz)
    os.environ["VF_DETECT_EVERY"] = str(args.detect_every)

    import uvicorn

    print(f"\n  Dashboard: http://{args.host}:{args.port}\n")
    uvicorn.run("api.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
