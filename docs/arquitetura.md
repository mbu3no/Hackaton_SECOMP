# Arquitetura

## Pipeline

```text
  CAPTURA                PROCESSAMENTO                      APRESENTAÇÃO
┌───────────┐   frame   ┌────────────────────────┐        ┌──────────────────┐
│ webcam    │ ────────► │ Detector (YOLOv8n)     │        │ /video_feed      │
│ celular   │           │   pessoas + pertences  │        │   MJPEG anotado  │
│ arquivo   │           └───────────┬────────────┘        │                  │
│ RTSP      │                       │ detecções           │ /api/state       │
└───────────┘                       ▼                     │   JSON, 500 ms   │
                        ┌────────────────────────┐        └────────┬─────────┘
                        │ Associação ROI ↔ det.  │                 │
                        │   IoS ≥ limiar         │                 ▼
                        └───────────┬────────────┘        ┌──────────────────┐
                                    ▼                     │ Dashboard        │
                        ┌────────────────────────┐        │   web/index.html │
                        │ Máquina de estados     │ ─────► │   contadores     │
                        │   por assento          │ estado │   alertas        │
                        │   + janela deslizante  │        └──────────────────┘
                        │   + cronômetro         │
                        └────────────────────────┘
```

`vision/pipeline.py` roda esse ciclo em uma thread dedicada. A API apenas lê o
último frame codificado e o estado atual, então uma inferência lenta degrada o
FPS do vídeo, mas nunca trava o servidor.

## Decisões técnicas

### 1. YOLOv8n pré-treinado, sem fine-tuning

Todas as classes necessárias já existem no COCO. Treinar não traria ganho
dentro da janela do evento e introduziria risco de não convergir. O `n` (nano)
foi escolhido para rodar em CPU, já que a demonstração é ao vivo no notebook.

### 2. Intersection over Smaller (IoS) no lugar do IoU

O IoU compara a interseção com a *união* das duas caixas. No nosso caso as
escalas são muito assimétricas:

- uma **pessoa em pé** gera uma caixa alta que engloba a ROI do assento → união enorme → IoU baixo;
- uma **garrafa** gera uma caixa minúscula dentro da ROI → união ≈ ROI → IoU baixo.

Nos dois casos o objeto *está* no assento, mas o IoU diria que não. Dividir
pela **menor** das duas áreas responde a pergunta certa: *"esta caixa está
contida naquela região?"*.

### 3. Estado temporal — a diferença em relação ao pipeline de referência

O exemplo distribuído pela organização é *stateless*: um frame entra, uma
resposta sai, nada é lembrado. Esse contrato não consegue expressar o nosso
problema, porque *"objeto sem dono há 20 minutos"* não existe em um frame
isolado.

Cada assento mantém:

- **janela deslizante** das últimas N observações, com voto majoritário — impede
  que uma detecção perdida em um ou dois frames faça o assento piscar;
- **cronômetro de abandono**, zerado sempre que uma pessoa é vista.

O período de tolerância é intencional: alguém que foi ao banheiro não deve
perder o lugar. Só depois dele a vaga é classificada como fantasma.

### 4. ROIs em coordenadas normalizadas

As ROIs são salvas em `[0..1]`, não em pixels. Assim a calibração feita em
1280×720 continua válida se a câmera abrir em outra resolução no dia do pitch.

### 5. MJPEG + JSON em vez de framework de dashboard

O navegador consome `<img src="/video_feed">` e faz *polling* de `/api/state`.
Isso mantém as três etapas do pipeline visivelmente separadas, não exige
WebRTC nem WebSocket, e o dashboard não depende de CDN — funciona offline.

## Parâmetros de operação

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--conf` | 0.35 | Confiança mínima do YOLO. Menor = mais detecções e mais falsos positivos |
| `--iou` | 0.25 | IoS mínimo para associar uma detecção a um assento |
| `--ghost-after` | 20 s | Tolerância antes de marcar como fantasma. **Em produção seriam minutos**; na demo é reduzido para caber no pitch |

<!-- TODO(equipe): registrar aqui os valores finais usados na demonstração. -->
