from ultralytics import YOLO

# Angiv stien til din model og data
model_path = "yolov8n.pt"  # Brug en pre-trænet YOLOv8 Nano-model
data_yaml = "C:/Users/Grams/OneDrive/Skrivebord/projectBow/ProjectBow.v1i.yolov11/data.yaml"

# Indlæs modellen
model = YOLO(model_path)

# Træn modellen
model.train(
    data=data_yaml,
    epochs=50,         # Antal træningsgennemløb
    imgsz=640,         # Billedstørrelse
    batch=8,           # Batch-størrelse
    workers=2          # Antal CPU-tråde til dataloading
)
