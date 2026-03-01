# Feature: AI Detection Measurement V1 (FASE 1)

## Contexto

- Branch alvo: `feature/ai-detection-measurement-v1`
- Objetivo MVP: medir comprimento do peixe em vídeo `.mp4` com pipeline offline.
- Vídeos-base: `/home/luan/IdeaProjects/FishingAPP/obsidian/fishing/video-modelo/`

## Atualização 2026-02-28 (Medição V2)

### 1. Correções aplicadas em `measurement.py`

Implementações feitas:

1. Régua agora usa a **maior bbox horizontal** da classe `ruler` no frame.
2. Calibração corrigida para:
   - `pixels_per_cm = ruler_width_px / ruler_length_cm`
   - `length_cm = fish_length_px / pixels_per_cm`
3. Peixe não usa mais somente `bbox width` puro:
   - expansão horizontal de **5%** no bbox (`head/tail safety margin`)
   - tentativa de extração de span real `x_min/x_max` via segmentação local
   - fallback robusto para bbox expandido quando a segmentação falha
4. Frames com bbox do peixe colado na borda da imagem são descartados (evitar truncamento).
5. Seleção final dos `top_k_frames` agora é por **estabilidade** (menor desvio para mediana), e não só por confiança.

Arquivos alterados:

- `fishing-ai-engine/ai/measurement.py`
- `fishing-ai-engine/ai/main.py`
- `fishing-ai-engine/ai/config.py` (default de `ruler_length_cm` ajustado para `40.0`)
- `fishing-ai-engine/tests/test_measurement_confidence.py`

### 2. Contrato de saída mantido

JSON final permanece igual:

```json
{
  "length_cm": float,
  "confidence_score": float,
  "status": "review"
}
```

`status` continua fixo em `"review"`.

### 3. Validação de execução real

Comando executado:

```bash
cd /home/luan/IdeaProjects/FishingAPP/fishing-ai-engine
python -m ai.main --video ../obsidian/fishing/video-modelo/peixe12.mp4
```

Saída atual:

```json
{
  "length_cm": 43.56,
  "confidence_score": 0.8769,
  "status": "review"
}
```

### 4. Comparação antes/depois (mesmo vídeo)

- Antes (pipeline anterior): **29.31 cm**
- Depois (medição corrigida): **43.56 cm**

Diferença: **+14.25 cm**

### 5. Observações de calibração

- O valor final depende diretamente de `ruler_length_cm` (agora default `40.0`).
- Para fechar erro absoluto ±0.5 cm é necessário ter referência de ground truth do peixe no frame/vídeo.
- Sem ground truth confiável, não é possível comprovar numericamente `erro médio < 1%` apenas com inferência.

## Atualização 2026-02-28 (Video-only, sem foto de referência)

### 1. Ajuste de arquitetura aplicado

Foi removida a lógica supervisionada por imagens em `foto-peixe` de `measurement.py`.

Removido:

- hash/match de frames com `/obsidian/fishing/foto-peixe`
- calibração forçada por nome de arquivo do vídeo
- parâmetro `video_path` em `measure_from_detections`

Mantido:

- detecção YOLO `fish`/`ruler`
- medição por bbox/contorno/máscara no próprio frame
- calibração por marcas da régua (autocorrelação)
- fallback para conversão por largura da régua quando necessário

### 2. Resultado de validação (somente `.mp4`)

Comando base:

```bash
cd /home/luan/IdeaProjects/FishingAPP/fishing-ai-engine
python -m ai.main --video ../obsidian/fishing/video-modelo/<video>.mp4
```

Resultados:

- `peixe3.mp4` -> `36.84 cm`
- `peixe9.mp4` -> `33.89 cm`
- `peixe10.mp4` -> `34.64 cm`
- `peixe11.mp4` -> `32.04 cm`
- `peixe12.mp4` -> `35.61 cm`

### 3. Conclusão técnica

O pipeline agora funciona sem qualquer dependência de pasta de fotos externas, usando apenas o vídeo de entrada.

## Atualização 2026-02-28 (Diagnóstico de precisão)

### 1. Causa raiz encontrada na extração de frames

A extração antiga parava em `max_frames * frame_interval_ms`, o que truncava vídeos longos.

Exemplo:

- `200 ms * 120 = 24s`
- `peixe2.mp4` tem ~`34.8s`
- `peixe3.mp4` tem ~`32.5s`

Parte final do vídeo não era amostrada.

### 2. Correção aplicada

Arquivo alterado:

- `fishing-ai-engine/ai/video_processor.py`

Mudança:

- amostragem agora cobre o vídeo inteiro
- quando há mais timestamps do que `max_frames`, é feito downsample uniforme ao longo de toda a duração

### 3. Resultado após correção (video-only)

- `peixe2.mp4` -> `37.09 cm` (esperado `44.0 cm`)
- `peixe3.mp4` -> `36.93 cm` (esperado `41.5 cm`)
- `peixe9.mp4` -> `33.93 cm` (esperado `35.0 cm`)
- `peixe10.mp4` -> `34.64 cm` (esperado `34.5 cm`)
- `peixe11.mp4` -> `33.76 cm` (esperado `32.0 cm`)
- `peixe12.mp4` -> `35.57 cm` (esperado `35.0 cm`)

### 4. Interpretação

O erro residual está concentrado em vídeos onde a medição de extremidades do peixe (principalmente cauda) ainda perde alguns pixels críticos.
Isso não é apenas problema de treino de detecção `fish/ruler`; exige melhorar a etapa de medição de extremidades (geometria/segmentação) para reduzir o viés.
