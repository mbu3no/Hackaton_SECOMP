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

### 3. Estado temporal (a diferença em relação ao pipeline de referência)

O exemplo distribuído pela organização é *stateless*: um frame entra, uma
resposta sai, nada é lembrado. Esse contrato não consegue expressar o nosso
problema, porque *"objeto sem dono há 20 minutos"* não existe em um frame
isolado.

Cada assento mantém:

- **janela deslizante** das últimas N observações, com voto majoritário, que impede
  que uma detecção perdida em um ou dois frames faça o assento piscar;
- **cronômetro de abandono**, zerado sempre que uma pessoa é vista.

O período de tolerância é intencional: alguém que foi ao banheiro não deve
perder o lugar. Só depois dele a vaga é classificada como fantasma.

### 4. ROIs em coordenadas normalizadas

As ROIs são salvas em `[0..1]`, não em pixels. Assim a calibração feita em
1280×720 continua válida se a câmera abrir em outra resolução no dia do pitch.

### 4b. Modo dinâmico (`--dynamic`): assentos sem calibração

Alternativa às ROIs fixas: em vez de zonas marcadas à mão, o YOLO detecta as
**cadeiras** (classes COCO `chair`, `bench`, `couch`) a cada ciclo, e um
rastreador leve (`vision/dynamic_seats.py`) casa cada detecção com o assento
conhecido mais próximo pelo centro. Vantagem: mover a cadeira ou a câmera não
quebra nada, e não há passo de calibração.

O desafio é a **oclusão**: quando uma pessoa senta, o corpo tapa a cadeira e o
YOLO deixa de detectá-la. Cada assento guarda a última posição e um par de
cronômetros (`_last_chair_seen`, `_last_alive`) e só é descartado quando a
cadeira sumiu **e** não há pessoa nem objeto por `seat_ttl_s` segundos. Assim
uma pessoa sentada mantém o assento vivo mesmo sem a cadeira aparecer.

### 5. Planta baixa por homografia

A câmera vê o chão em perspectiva: um retângulo real vira um trapézio na
imagem. `cv2.getPerspectiveTransform` sobre 4 cantos conhecidos devolve a
matriz que desfaz isso, e `cv2.perspectiveTransform` projeta qualquer ponto.

Duas consequências práticas:

- **Só o plano do chão é válido.** Cadeira e pessoa têm altura, então projetar
  a caixa inteira dá erro. Projetamos sempre a **base** da caixa, o ponto em
  que o objeto encosta no chão (`FloorPlan.ground_anchor`).
- **Os assentos não são posicionados à mão no mapa.** A posição sai da
  projeção da ROI, então a planta acompanha a calibração automaticamente.

A calibração é opcional: sem `config/homography.json` o sistema roda igual e o
dashboard apenas esconde o painel.

### 6. MJPEG + JSON em vez de framework de dashboard

O navegador consome `<img src="/video_feed">` e faz *polling* de `/api/state`.
Isso mantém as três etapas do pipeline visivelmente separadas, não exige
WebRTC nem WebSocket, e o dashboard não depende de CDN, e funciona offline.

## Parâmetros de operação

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--conf` | 0.35 | Confiança mínima do YOLO. Menor = mais detecções e mais falsos positivos |
| `--iou` | 0.25 | IoS mínimo para associar uma detecção a um assento |
| `--homography` | `config/homography.json` | Calibração da planta. Ausente = painel escondido |
| `--ghost-after` | 20 s | Tolerância antes de marcar como fantasma. **Em produção seriam minutos**; na demo é reduzido para caber no pitch |

<!-- TODO(equipe): registrar aqui os valores finais usados na demonstração. -->
