

rodar por video
python -m ai.main --video ../obsidian/fishing/video-modelo/peixe12.mp4


rodar modelo para treinar
python /home/luan/IdeaProjects/FishingAPP/fishing-ai-engine/scripts/train_yolo.py \
    --model yolov8s.pt \
    --epochs 80 \
    --imgsz 640 \
    --batch 16 \
    --name fishing_ai_v2
