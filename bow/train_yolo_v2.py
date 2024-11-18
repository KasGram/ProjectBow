from ultralytics import YOLO

# Sti til modellen og data
model_path = "yolov8m.pt"  # Brug en pre-trænet YOLOv8 Nano-model
data_yaml = "C:/Users/Grams/OneDrive/Skrivebord/projectBow/ProjectBow.v1i.yolov11/data.yaml"

# Indlæs YOLO-modellen
model = YOLO(model_path)

# Første træningsrunde med 100 epochs
model.train(
    data=data_yaml,       # Data-konfigurationsfil
    epochs=100,           # Antal trænings-epochs
    imgsz=640,            # Input-billedstørrelse
    batch=8,              # Batch-størrelse
    workers=2,            # Antal CPU-tråde til dataloading
    patience=20           # Early stopping efter 20 epochs uden forbedring
)

# Gemmer modellen fra første træningsrunde
trained_model_path = "runs/detect/train/weights/best.pt"  # Juster stien afhængigt af output
print(f"Model gemt: {trained_model_path}")

# Indlæser den tidligere trænet model for yderligere træning
model = YOLO(trained_model_path)

# Validering for at evaluere første træningsresultat
results = model.val(
    data=data_yaml,       # Data-konfigurationsfil
    imgsz=640,            # Samme billedstørrelse
    batch=8               # Batch-størrelse
)
print(f"Resultater efter første træning: {results}")

# Finjustering (videre træning) med yderligere 100 epochs
model.train(
    data=data_yaml,       # Data-konfigurationsfil
    epochs=100,           # Flere trænings-epochs
    imgsz=640,            # Input-billedstørrelse
    batch=8,              # Batch-størrelse
    workers=2,            # Antal CPU-tråde
    freeze=10             # Fryser de første 10 lag
)

# Gemmer den finjusterede model
fine_tuned_model_path = "runs/detect/train2/weights/best.pt"  # Output for videre træning
print(f"Finjusteret model gemt: {fine_tuned_model_path}")

# Test og evaluering efter finjustering
results = model.val(
    data=data_yaml,       # Data-konfigurationsfil
    imgsz=640,            # Samme billedstørrelse
    batch=8               # Batch-størrelse
)

# Udskriver evalueringens resultater
print(f"Evaluering færdig efter finjustering: {results}")
