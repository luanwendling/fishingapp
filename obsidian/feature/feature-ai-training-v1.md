# Feature: AI Training V1 (YOLOv8 fish + ruler)

## Objetivo

Treinar um modelo YOLOv8 para detectar:

- `fish` (classe 0)
- `ruler` (classe 1)

Output alvo do modelo:

- `fishing-ai-engine/ai/models/best.pt`

## Atualização 2026-02-28 (V2)

### 1. Dataset corrigido (todos os 12 vídeos + amostragem no vídeo inteiro)

Fonte:

- `/home/luan/IdeaProjects/FishingAPP/obsidian/fishing/video-modelo/peixe1.mp4 ... peixe12.mp4`

Estratégia aplicada:

1. Extração distribuída ao longo de todo vídeo com intervalo fixo (`250ms`).
2. Pseudo-label com modelo existente (`ai/models/best.pt`) + fallback heurístico.
3. Filtros de validade (fish/ruler, overlap mínimo, áreas mínimas).
4. Split automático train/val (80/20).

Comando usado:

```bash
python fishing-ai-engine/scripts/build_dataset.py \
  --clean \
  --frame-interval-ms 250 \
  --fish-conf-min 0.22 \
  --ruler-conf-min 0.20
```

Resultado:

- `Videos found: 12 | Videos used: 12`
- Total imagens: **729**
- Train: **584**
- Val: **145**

Contagem por vídeo:

- peixe1: 25
- peixe2: 67
- peixe3: 91
- peixe4: 59
- peixe5: 46
- peixe6: 55
- peixe7: 71
- peixe8: 73
- peixe9: 69
- peixe10: 44
- peixe11: 66
- peixe12: 63

### 2. Treino V2 com YOLOv8s

Configuração solicitada:

- `model=yolov8s.pt`
- `epochs=80`
- `imgsz=640`
- `batch=16`
- `name=fishing_ai_v2`

Comando executado:

```bash
python fishing-ai-engine/scripts/train_yolo.py \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --name fishing_ai_v2
```

Status:

- Execução iniciada e em progresso, porém interrompida manualmente por tempo de runtime em CPU.
- Melhor checkpoint parcial foi salvo e promovido para:
  - `fishing-ai-engine/ai/models/best.pt`

### 3. Métricas atuais do melhor checkpoint V2

Validação do checkpoint atual (`best.pt`) em `dataset/data.yaml`:

- mAP50 (geral): **0.669**
- mAP50-95 (geral): **0.427**
- Precision: **0.632**
- Recall: **0.646**

Por classe (`class_result`):

- fish: **mAP50 = 0.503** | mAP50-95 = **0.288**
- ruler: **mAP50 = 0.835** | mAP50-95 = **0.565**

### 4. Validação de detecção em vídeo real (`peixe12.mp4`)

Com o novo `best.pt`:

- `conf=0.25`: fish **38/39**, ruler **39/39**, ambos **38/39**
- `conf=0.10`: fish **38/39**, ruler **39/39**, ambos **38/39**
- `conf=0.05`: fish **38/39**, ruler **39/39**, ambos **38/39**

### 5. Gap para meta final

Metas pedidas:

- fish mAP50 > 0.75
- ruler mAP50 > 0.90

Status atual:

- fish mAP50 **0.503** (ainda abaixo)
- ruler mAP50 **0.835** (ainda abaixo)

### 6. Próximo passo recomendado para fechar a meta

Para concluir exatamente as metas, manter o treino V2 até completar as 80 épocas e revalidar:

```bash
python fishing-ai-engine/scripts/train_yolo.py \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --name fishing_ai_v2
```

Depois revalidar:

```bash
python -m ai.main --video ../obsidian/fishing/video-modelo/peixe12.mp4
```
