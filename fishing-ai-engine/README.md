# Fishing AI Engine (FASE 1 MVP)

Pipeline offline em Python para medir comprimento de peixe em vídeo `.mp4` com YOLOv8.

## Estrutura

```text
fishing-ai-engine/
├── ai/
│   ├── __init__.py
│   ├── main.py
│   ├── video_processor.py
│   ├── detector.py
│   ├── measurement.py
│   ├── confidence.py
│   ├── utils.py
│   ├── config.py
│   └── models/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Requisitos

- Python 3.10+
- Modelo YOLOv8 treinado com classes `fish` e `ruler`

Instalação:

```bash
pip install -r requirements.txt
```

## Treinamento do modelo

Estrutura de dataset esperada no root do projeto (`/home/luan/IdeaProjects/FishingAPP/dataset`):

```text
dataset/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

Scripts de apoio (em `fishing-ai-engine/scripts`):

```bash
# 1) Gera dataset a partir dos 12 vídeos (amostragem ao longo do vídeo inteiro)
python scripts/build_dataset.py --clean --frame-interval-ms 250 --fish-conf-min 0.22 --ruler-conf-min 0.20

# 2) Treina YOLO e copia o best.pt para ai/models/best.pt
python scripts/train_yolo.py --model yolov8s.pt --epochs 80 --imgsz 640 --batch 16 --name fishing_ai_v2

# 3) Valida em vídeo real
python scripts/predict_video.py \
  --model ai/models/best.pt \
  --source /home/luan/IdeaProjects/FishingAPP/obsidian/fishing/video-modelo/peixe12.mp4
```

## Execução

No diretório `fishing-ai-engine`:

```bash
python -m ai.main --video /caminho/video.mp4 --model ai/models/best.pt
```

Parâmetros úteis:

- `--frame-interval-ms` (default: `200`)
- `--max-frames` (default: `120`)
- `--ruler-length-cm` (default: `40.0`)
- `--top-k-frames` (default: `5`)

## Saída JSON (MVP)

```json
{
  "length_cm": 32.4,
  "confidence_score": 0.87,
  "status": "review"
}
```

## Observações

- Nesta fase o status sempre é `review`.
- Não inclui OCR, perspectiva, antifraude, backend ou mobile.
- Se o modelo não estiver em `ai/models/best.pt`, informe `--model` manualmente.
